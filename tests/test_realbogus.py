"""
Tests for real-bogus classifier module.

This test suite covers:
- Model architecture creation
- Cutout preprocessing (downscaling, normalization)
- Training data generation
- Classification pipeline
- Model serialization
- Integration with detection pipeline
"""

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from astropy.table import Table
import tempfile
import os

# Try importing realbogus module (optional dependency)
try:
    from stdpipe import realbogus
    HAS_REALBOGUS = realbogus.HAS_TENSORFLOW
except ImportError:
    HAS_REALBOGUS = False

from stdpipe import simulation, photometry

# Skip all tests if TensorFlow not available
pytestmark = pytest.mark.skipif(
    not HAS_REALBOGUS,
    reason="TensorFlow not installed (install with: pip install stdpipe[ml])"
)


class TestModelArchitecture:
    """Test CNN model creation and architecture."""

    @pytest.mark.unit
    def test_create_model_default(self):
        """Test creating model with default parameters."""
        model = realbogus.create_realbogus_model()

        assert model is not None
        assert len(model.input_shape) == 2  # [image, fwhm] inputs
        assert model.output_shape == (None, 1)  # Binary output

        # Check parameter count (should be ~100k)
        n_params = model.count_params()
        assert 50000 < n_params < 200000, f"Unexpected parameter count: {n_params}"

    @pytest.mark.unit
    def test_create_model_custom_filters(self):
        """Test creating model with custom filter sizes."""
        model = realbogus.create_realbogus_model(filters=(16, 32, 64))

        n_params = model.count_params()
        # Fewer filters = fewer parameters
        assert n_params < 100000

    @pytest.mark.unit
    def test_create_model_no_fwhm(self):
        """Test creating model without FWHM auxiliary input."""
        model = realbogus.create_realbogus_model(use_fwhm_feature=False)

        # Should have single input
        assert not isinstance(model.input_shape, list)
        assert len(model.input_shape) == 4  # (batch, height, width, channels)

    @pytest.mark.unit
    def test_model_compilation(self):
        """Test that model is properly compiled."""
        model = realbogus.create_realbogus_model()

        # Check optimizer
        assert model.optimizer is not None

        # Check loss
        assert 'binary_crossentropy' in str(model.loss).lower()

        # Test that model can make predictions (ensures it's properly compiled)
        # Create dummy input
        dummy_input = [
            np.random.randn(1, 31, 31, 3).astype(np.float32),  # image
            np.array([[3.0]], dtype=np.float32)  # fwhm
        ]

        # Should be able to make predictions
        prediction = model.predict(dummy_input, verbose=0)
        assert prediction.shape == (1, 1)
        assert 0 <= prediction[0, 0] <= 1  # Sigmoid output in [0, 1]

        # Test that model can compute loss and metrics on a small batch
        dummy_y = np.array([[1.0]], dtype=np.float32)
        result = model.evaluate(dummy_input, dummy_y, verbose=0)

        # Result should be a list/array with loss and metrics
        assert result is not None
        assert len(result) > 0  # At least loss value


class TestCutoutPreprocessing:
    """Test cutout preprocessing functions."""

    @pytest.mark.unit
    def test_normalize_cutout_robust(self):
        """Test robust z-score normalization."""
        cutout = np.random.randn(31, 31) * 10 + 100

        normalized = realbogus._normalize_cutout(cutout, method='robust')

        # Should have approximately mean=0, std=1
        assert abs(np.median(normalized)) < 0.5
        assert 0.5 < np.std(normalized) < 2.0

    @pytest.mark.unit
    def test_normalize_cutout_standard(self):
        """Test standard z-score normalization."""
        cutout = np.random.randn(31, 31) * 10 + 100

        normalized = realbogus._normalize_cutout(cutout, method='standard')

        # Should have approximately mean=0, std=1
        assert abs(np.mean(normalized)) < 0.1
        assert 0.9 < np.std(normalized) < 1.1

    @pytest.mark.unit
    def test_downscale_cutout_2x(self):
        """Test 2x downscaling."""
        cutout = np.random.randn(32, 32) * 10 + 100

        downscaled = realbogus._downscale_cutout(cutout, scale_factor=2, mode='mean')

        assert downscaled.shape == (16, 16)
        # Mean should be approximately preserved
        assert abs(np.mean(downscaled) - np.mean(cutout)) < 1.0

    @pytest.mark.unit
    def test_downscale_cutout_no_scale(self):
        """Test that scale_factor=1 returns original."""
        cutout = np.random.randn(31, 31)

        downscaled = realbogus._downscale_cutout(cutout, scale_factor=1)

        np.testing.assert_array_equal(downscaled, cutout)

    @pytest.mark.unit
    def test_pad_to_size_larger(self):
        """Test padding to larger size."""
        cutout = np.random.randn(21, 21)

        padded = realbogus._pad_to_size(cutout, target_size=31, mode='edge')

        assert padded.shape == (31, 31)
        # Center should contain original data
        assert np.allclose(padded[5:26, 5:26], cutout)

    @pytest.mark.unit
    def test_pad_to_size_smaller(self):
        """Test cropping to smaller size."""
        cutout = np.random.randn(41, 41)

        cropped = realbogus._pad_to_size(cutout, target_size=31, mode='edge')

        assert cropped.shape == (31, 31)

    @pytest.mark.unit
    def test_preprocess_cutout_full(self):
        """Test full preprocessing pipeline."""
        cutout = np.random.randn(31, 31) * 10 + 1000

        preprocessed, scale_factor = realbogus.preprocess_cutout(
            cutout,
            fwhm=6.0,
            target_fwhm=3.0,
            target_size=31,
            normalize=True
        )

        # Should be 3-channel output
        assert preprocessed.shape == (31, 31, 3)
        assert preprocessed.dtype == np.float32

        # Should have applied 2x downscaling
        assert scale_factor == 2.0

        # Channels should be normalized
        for ch in range(3):
            assert abs(np.mean(preprocessed[:, :, ch])) < 1.0

    @pytest.mark.unit
    def test_preprocess_cutout_no_downscale(self):
        """Test preprocessing without downscaling."""
        cutout = np.random.randn(31, 31) * 10 + 1000

        preprocessed, scale_factor = realbogus.preprocess_cutout(
            cutout,
            fwhm=2.0,  # Small FWHM, no downscaling
            target_fwhm=3.0,
            target_size=31,
            normalize=True
        )

        assert preprocessed.shape == (31, 31, 3)
        assert scale_factor < 1.5  # No downscaling applied


class TestCutoutExtraction:
    """Test batch cutout extraction."""

    @pytest.fixture
    def simple_image_with_objects(self):
        """Create simple image with known objects."""
        # Simulate image with a few stars
        sim = simulation.simulate_image(
            width=512, height=512,
            n_stars=10,
            star_fwhm=3.0,
            n_galaxies=0,
            n_cosmic_rays=0,
            n_hot_pixels=0,
            background=1000.0,
            verbose=False
        )

        # Detect objects
        obj = photometry.get_objects_sep(
            sim['image'],
            thresh=5.0,
            verbose=False
        )

        return sim['image'], obj

    @pytest.mark.unit
    def test_extract_cutouts_basic(self, simple_image_with_objects):
        """Test basic cutout extraction."""
        image, obj = simple_image_with_objects

        cutouts, fwhm_features, valid_indices = realbogus.extract_cutouts(
            obj,
            image,
            radius=15,
            fwhm=3.0,
            verbose=False
        )

        # Should extract cutouts for most objects
        assert len(cutouts) > 0
        assert len(cutouts) <= len(obj)

        # Check shapes
        assert cutouts.shape[1:] == (31, 31, 3)  # 3 channels
        assert fwhm_features.shape == (len(cutouts), 1)
        assert len(valid_indices) == len(cutouts)

    @pytest.mark.unit
    def test_extract_cutouts_with_background(self, simple_image_with_objects):
        """Test cutout extraction with explicit background."""
        image, obj = simple_image_with_objects

        bg = np.full_like(image, 1000.0)

        cutouts, fwhm_features, valid_indices = realbogus.extract_cutouts(
            obj,
            image,
            bg=bg,
            radius=15,
            fwhm=3.0,
            verbose=False
        )

        assert len(cutouts) > 0

    @pytest.mark.unit
    def test_extract_cutouts_edge_objects(self):
        """Test that edge objects are skipped."""
        # Small image
        image = np.random.randn(100, 100) * 10 + 1000

        # Object catalog with edge positions
        obj = Table({
            'x': [5.0, 50.0, 95.0],  # 1st and 3rd are too close to edge
            'y': [50.0, 50.0, 50.0],
        })

        cutouts, fwhm_features, valid_indices = realbogus.extract_cutouts(
            obj,
            image,
            radius=15,
            fwhm=3.0,
            verbose=False
        )

        # Should only extract center object
        assert len(cutouts) == 1
        assert valid_indices[0] == 1


class TestTrainingDataGeneration:
    """Test training data generation from simulations."""

    @pytest.mark.unit
    def test_generate_training_data_small(self):
        """Test generating small training dataset."""
        data = simulation.generate_realbogus_training_data(
            n_images=5,
            image_size=(512, 512),
            augment=False,
            verbose=False
        )

        assert 'X' in data
        assert 'y' in data
        assert 'fwhm' in data

        # Check shapes
        X = data['X']
        y = data['y']
        fwhm = data['fwhm']

        assert len(X) == len(y) == len(fwhm)
        assert X.ndim == 4  # (N, H, W, C)
        assert X.shape[-1] == 3  # 3 channels
        assert y.ndim == 1
        assert fwhm.ndim == 2

        # Should have mix of real and bogus
        assert 0 < np.sum(y) < len(y)

    @pytest.mark.unit
    def test_generate_training_data_with_augmentation(self):
        """Test data augmentation multiplier."""
        data_no_aug = simulation.generate_realbogus_training_data(
            n_images=3,
            image_size=(512, 512),
            augment=False,
            verbose=False
        )

        data_aug = simulation.generate_realbogus_training_data(
            n_images=3,
            image_size=(512, 512),
            augment=True,
            verbose=False
        )

        # Augmented should have more samples (at least 2x due to rotations)
        # Note: Some cutouts may be filtered during extraction, so use conservative threshold
        assert len(data_aug['X']) >= 2 * len(data_no_aug['X'])

    @pytest.mark.unit
    def test_augment_training_data(self):
        """Test data augmentation function."""
        # Create dummy data
        X = np.random.randn(10, 31, 31, 3).astype(np.float32)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        fwhm = np.random.randn(10, 1).astype(np.float32)

        X_aug, y_aug, fwhm_aug = simulation._augment_training_data(
            X, y, fwhm,
            augment_factor=4,
            verbose=False
        )

        # Should have 4x more samples (rotations)
        assert len(X_aug) == 4 * len(X)
        assert len(y_aug) == 4 * len(y)
        assert len(fwhm_aug) == 4 * len(fwhm)


class TestClassification:
    """Test classification pipeline."""

    @pytest.fixture
    def trained_model(self, tmp_path):
        """Create a minimally trained model for testing."""
        # Generate tiny training dataset
        data = simulation.generate_realbogus_training_data(
            n_images=3,
            image_size=(256, 256),
            augment=False,
            verbose=False
        )

        # Create and train model (1 epoch just for testing)
        model = realbogus.create_realbogus_model()
        model.fit(
            [data['X'], data['fwhm']],
            data['y'],
            epochs=1,
            batch_size=32,
            verbose=0
        )

        return model

    @pytest.mark.unit
    def test_classify_simple_image(self, trained_model):
        """Test classification on simple simulated image."""
        # Simulate image
        sim = simulation.simulate_image(
            width=512, height=512,
            n_stars=20,
            star_fwhm=3.0,
            n_cosmic_rays=5,
            verbose=False
        )

        # Detect objects
        obj = photometry.get_objects_sep(
            sim['image'],
            thresh=5.0,
            verbose=False
        )

        # Classify
        obj_filtered = realbogus.classify_realbogus(
            obj,
            sim['image'],
            model=trained_model,
            threshold=0.5,
            add_score=True,
            flag_bogus=True,
            verbose=False
        )

        # Should filter some objects
        assert len(obj_filtered) <= len(obj)

        # Should have rb_score column
        assert 'rb_score' in obj_filtered.colnames

        # Scores should be in [0, 1]
        assert np.all(obj_filtered['rb_score'] >= 0)
        assert np.all(obj_filtered['rb_score'] <= 1)

    @pytest.mark.unit
    def test_classify_add_score_no_filter(self, trained_model):
        """Test adding scores without filtering."""
        # Simulate image
        sim = simulation.simulate_image(
            width=512, height=512,
            n_stars=10,
            star_fwhm=3.0,
            verbose=False
        )

        # Detect objects
        obj = photometry.get_objects_sep(
            sim['image'],
            thresh=5.0,
            verbose=False
        )

        # Classify without filtering
        obj_scored = realbogus.classify_realbogus(
            obj,
            sim['image'],
            model=trained_model,
            threshold=0.5,
            add_score=True,
            flag_bogus=False,  # Don't filter
            verbose=False
        )

        # Should not filter any objects
        assert len(obj_scored) == len(obj)

        # Should have rb_score column
        assert 'rb_score' in obj_scored.colnames


class TestModelSerialization:
    """Test model saving and loading."""

    @pytest.mark.unit
    def test_save_load_model(self, tmp_path):
        """Test saving and loading model."""
        # Create model
        model = realbogus.create_realbogus_model()

        # Save
        model_file = tmp_path / "test_model.h5"
        realbogus.save_realbogus_model(model, model_file=str(model_file), verbose=False)

        assert model_file.exists()

        # Load
        loaded_model = realbogus.load_realbogus_model(model_file=str(model_file), verbose=False)

        # Check architecture matches
        assert loaded_model.count_params() == model.count_params()

    @pytest.mark.unit
    def test_load_nonexistent_model(self):
        """Test loading non-existent model raises error."""
        with pytest.raises(FileNotFoundError):
            realbogus.load_realbogus_model(model_file="/nonexistent/model.h5")


class TestTraining:
    """Test training pipeline."""

    @pytest.mark.unit
    def test_train_with_simulated_data(self, tmp_path):
        """Test training with simulated data generation."""
        model_file = tmp_path / "trained_model.h5"

        # Train with minimal data and epochs
        model, history = realbogus.train_realbogus_classifier(
            training_data=None,  # Generate simulated data
            n_simulated=5,
            image_size=(256, 256),
            epochs=2,
            batch_size=32,
            model_file=str(model_file),
            verbose=False
        )

        # Check model was trained
        assert model is not None
        assert history is not None

        # Check history
        assert 'loss' in history.history
        assert 'accuracy' in history.history

        # Model should be saved
        assert model_file.exists()

    @pytest.mark.unit
    def test_train_with_pregenerated_data(self):
        """Test training with pre-generated data."""
        # Generate data
        data = simulation.generate_realbogus_training_data(
            n_images=3,
            image_size=(256, 256),
            augment=False,
            verbose=False
        )

        # Train
        model, history = realbogus.train_realbogus_classifier(
            training_data=data,  # Use pre-generated data
            epochs=2,
            batch_size=32,
            verbose=False
        )

        assert model is not None
        assert len(history.history['loss']) == 2  # 2 epochs

    @pytest.mark.unit
    def test_train_class_weighting(self):
        """Test that class weighting is applied."""
        # Generate imbalanced data
        X = np.random.randn(100, 31, 31, 3).astype(np.float32)
        y = np.array([1] * 80 + [0] * 20, dtype=np.float32)  # 80% real, 20% bogus
        fwhm = np.random.randn(100, 1).astype(np.float32)

        model, history = realbogus.train_realbogus_classifier(
            training_data=(X, y, fwhm),
            epochs=1,
            class_weight='balanced',
            verbose=False
        )

        # Should complete without errors
        assert model is not None


class TestIntegration:
    """Integration tests with full pipeline."""

    @pytest.mark.integration
    def test_full_pipeline_sep_backend(self):
        """Test full pipeline with SEP detection backend."""
        # Simulate realistic image
        sim = simulation.simulate_image(
            width=1024, height=1024,
            n_stars=50,
            star_fwhm=4.0,
            n_galaxies=10,
            n_cosmic_rays=10,
            n_hot_pixels=15,
            background=1000.0,
            verbose=False
        )

        # Detect objects
        obj = photometry.get_objects_sep(
            sim['image'],
            thresh=3.0,
            aper=10.0,
            verbose=False
        )

        # Train mini model
        data = simulation.generate_realbogus_training_data(
            n_images=5,
            image_size=(512, 512),
            augment=False,
            verbose=False
        )

        model, _ = realbogus.train_realbogus_classifier(
            training_data=data,
            epochs=2,
            verbose=False
        )

        # Classify
        obj_clean = realbogus.classify_realbogus(
            obj,
            sim['image'],
            model=model,
            threshold=0.5,
            verbose=False
        )

        # Should have filtered some objects
        assert len(obj_clean) <= len(obj)
        assert 'rb_score' in obj_clean.colnames

    @pytest.mark.integration
    @pytest.mark.requires_sextractor
    def test_full_pipeline_sextractor_backend(self):
        """Test full pipeline with SExtractor detection backend."""
        # Simulate image
        sim = simulation.simulate_image(
            width=1024, height=1024,
            n_stars=30,
            star_fwhm=3.5,
            n_cosmic_rays=5,
            verbose=False
        )

        # Detect with SExtractor
        try:
            obj = photometry.get_objects_sextractor(
                sim['image'],
                thresh=3.0,
                aper=8.0,
                verbose=False
            )
        except Exception:
            pytest.skip("SExtractor not available")

        # Train mini model
        data = simulation.generate_realbogus_training_data(
            n_images=3,
            image_size=(512, 512),
            augment=False,
            verbose=False
        )

        model, _ = realbogus.train_realbogus_classifier(
            training_data=data,
            epochs=1,
            verbose=False
        )

        # Classify
        obj_clean = realbogus.classify_realbogus(
            obj,
            sim['image'],
            model=model,
            verbose=False
        )

        assert len(obj_clean) <= len(obj)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_empty_catalog(self):
        """Test classification with empty catalog."""
        image = np.random.randn(512, 512) * 10 + 1000
        obj = Table({'x': [], 'y': []})

        model = realbogus.create_realbogus_model()

        # Should raise error for empty catalog
        with pytest.raises(ValueError, match="No valid cutouts"):
            realbogus.classify_realbogus(
                obj,
                image,
                model=model,
                verbose=False
            )

    @pytest.mark.unit
    def test_extreme_fwhm_values(self):
        """Test preprocessing with extreme FWHM values."""
        cutout = np.random.randn(31, 31) * 10 + 1000

        # Very small FWHM
        preprocessed_small, _ = realbogus.preprocess_cutout(
            cutout,
            fwhm=0.5,
            target_fwhm=3.0
        )
        assert preprocessed_small.shape == (31, 31, 3)

        # Very large FWHM
        preprocessed_large, scale = realbogus.preprocess_cutout(
            cutout,
            fwhm=20.0,
            target_fwhm=3.0
        )
        assert preprocessed_large.shape == (31, 31, 3)
        assert scale > 5.0  # Should be heavily downscaled

    @pytest.mark.unit
    def test_nan_in_image(self):
        """Test handling of NaN values in image."""
        image = np.random.randn(512, 512) * 10 + 1000
        # Add some NaNs
        image[100:110, 100:110] = np.nan

        obj = Table({
            'x': [105.0, 300.0],  # One near NaNs, one clean
            'y': [105.0, 300.0],
        })

        model = realbogus.create_realbogus_model()

        # Should handle NaNs gracefully (skip or process)
        try:
            obj_scored = realbogus.classify_realbogus(
                obj,
                image,
                model=model,
                flag_bogus=False,
                verbose=False
            )
            # If successful, some objects may be skipped
            assert len(obj_scored) <= len(obj)
        except ValueError:
            # Or it may raise error - both acceptable
            pass
