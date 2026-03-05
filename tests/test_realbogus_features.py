"""
Tests for feature-based real-bogus classification.

Tests the realbogus_features module which provides classification
without requiring TensorFlow.
"""

import numpy as np
import pytest
from astropy.table import Table

from stdpipe import realbogus_features as rbf


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_catalog():
    """Create a simple object catalog."""
    n = 50
    rng = np.random.RandomState(42)

    obj = Table()
    obj['x'] = rng.uniform(50, 450, n)
    obj['y'] = rng.uniform(50, 450, n)
    obj['flux'] = rng.uniform(1000, 10000, n)
    obj['fluxerr'] = rng.uniform(10, 100, n)
    obj['fwhm'] = rng.normal(3.0, 0.3, n)
    obj['a'] = rng.uniform(1.5, 2.5, n)
    obj['b'] = rng.uniform(1.3, 2.0, n)
    obj['peak'] = obj['flux'] / 10
    obj['FLUX_RADIUS'] = obj['fwhm'] * 0.8
    obj['FLUX_MAX'] = obj['peak']
    obj['FLUX_AUTO'] = obj['flux']
    obj['flags'] = np.zeros(n, dtype=np.int32)

    return obj


@pytest.fixture
def simple_image():
    """Create a simple test image with point sources."""
    rng = np.random.RandomState(42)
    size = 512

    # Background + noise
    image = np.ones((size, size)) * 1000 + rng.normal(0, 10, (size, size))

    # Add some point sources (Gaussian)
    fwhm = 3.0
    sigma = fwhm / 2.355

    for x, y, flux in [(100, 100, 5000), (200, 200, 8000), (300, 300, 3000),
                       (400, 400, 6000), (150, 350, 4000)]:
        yy, xx = np.ogrid[:size, :size]
        r2 = (xx - x)**2 + (yy - y)**2
        psf = np.exp(-r2 / (2 * sigma**2))
        image += flux * psf / psf.sum()

    # Add a cosmic ray (sharp spike)
    image[250, 250] += 10000

    # Add a hot pixel
    image[350, 150] += 5000

    return image.astype(np.float32)


@pytest.fixture
def catalog_with_sources(simple_image):
    """Create catalog matching sources in simple_image."""
    obj = Table()
    # Real sources
    obj['x'] = [100, 200, 300, 400, 150]
    obj['y'] = [100, 200, 300, 400, 350]
    obj['flux'] = [5000, 8000, 3000, 6000, 4000]
    obj['fluxerr'] = [50, 80, 30, 60, 40]
    obj['fwhm'] = [3.0, 3.0, 3.0, 3.0, 3.0]
    obj['a'] = [1.5, 1.5, 1.5, 1.5, 1.5]
    obj['b'] = [1.4, 1.4, 1.4, 1.4, 1.4]
    obj['peak'] = [500, 800, 300, 600, 400]
    obj['flags'] = np.zeros(5, dtype=np.int32)

    # Add cosmic ray and hot pixel
    obj.add_row({'x': 250, 'y': 250, 'flux': 10000, 'fluxerr': 100,
                 'fwhm': 1.0, 'a': 0.5, 'b': 0.5, 'peak': 10000, 'flags': 0})
    obj.add_row({'x': 150, 'y': 350, 'flux': 5000, 'fluxerr': 50,
                 'fwhm': 0.5, 'a': 0.3, 'b': 0.3, 'peak': 5000, 'flags': 0})

    return obj


# =============================================================================
# Tests: Feature Extraction
# =============================================================================

class TestCatalogFeatures:
    """Tests for catalog-based feature extraction."""

    @pytest.mark.unit
    def test_extract_basic_features(self, simple_catalog):
        """Test basic catalog feature extraction."""
        features, names = rbf.extract_catalog_features(simple_catalog)

        assert isinstance(features, dict)
        assert len(names) > 0
        assert 'fwhm' in features
        assert len(features['fwhm']) == len(simple_catalog)

    @pytest.mark.unit
    def test_extract_ellipticity(self, simple_catalog):
        """Test ellipticity calculation."""
        features, names = rbf.extract_catalog_features(simple_catalog)

        assert 'ellipticity' in features
        # Ellipticity = (a-b)/(a+b) can be negative if b > a in random data
        # In real data a >= b, but we just check it's bounded
        ell = features['ellipticity']
        assert np.all((ell >= -1) & (ell <= 1) | ~np.isfinite(ell))

    @pytest.mark.unit
    def test_extract_fwhm_ratio(self, simple_catalog):
        """Test FWHM ratio calculation."""
        features, names = rbf.extract_catalog_features(simple_catalog)

        assert 'fwhm_ratio' in features
        # Ratio should be close to 1 for typical sources
        ratio = features['fwhm_ratio']
        assert np.nanmedian(ratio) == pytest.approx(1.0, rel=0.1)

    @pytest.mark.unit
    def test_invalid_feature_set_raises(self, simple_catalog):
        """Invalid feature set should raise a clear error."""
        with pytest.raises(ValueError, match="Unknown feature_set"):
            rbf.extract_catalog_features(simple_catalog, feature_set='invalid')


class TestCutoutFeatures:
    """Tests for cutout-based feature extraction."""

    @pytest.mark.unit
    def test_extract_cutout_features(self, catalog_with_sources, simple_image):
        """Test cutout feature extraction."""
        features, names = rbf.extract_cutout_features(
            catalog_with_sources, simple_image, fwhm=3.0, radius=10
        )

        assert isinstance(features, dict)
        assert 'sharpness' in features
        assert 'roundness' in features
        assert 'symmetry' in features
        assert 'concentration' in features
        assert len(features['sharpness']) == len(catalog_with_sources)

    @pytest.mark.unit
    def test_sharpness_detects_cosmic_ray(self, catalog_with_sources, simple_image):
        """Test that cosmic rays have high sharpness."""
        features, _ = rbf.extract_cutout_features(
            catalog_with_sources, simple_image, fwhm=3.0, radius=10
        )

        # Cosmic ray is at index 5 (last two are artifacts)
        sharpness = features['sharpness']
        real_sharpness = np.nanmean(sharpness[:5])
        cr_sharpness = sharpness[5]  # Cosmic ray

        # Cosmic ray should be sharper
        assert cr_sharpness > real_sharpness

    @pytest.mark.unit
    def test_roundness_detects_elongated(self, catalog_with_sources, simple_image):
        """Test roundness calculation."""
        features, _ = rbf.extract_cutout_features(
            catalog_with_sources, simple_image, fwhm=3.0, radius=10
        )

        roundness = features['roundness']
        # Real sources should be reasonably round
        assert np.nanmean(roundness[:5]) > 0.5

    @pytest.mark.unit
    def test_cutout_snr_computed_when_err_provided(self, catalog_with_sources, simple_image):
        """Error map input should produce cutout_snr feature."""
        features, _ = rbf.extract_cutout_features(
            catalog_with_sources, simple_image, fwhm=3.0, radius=10, err=10.0
        )
        assert 'cutout_snr' in features
        assert np.any(np.isfinite(features['cutout_snr']))


class TestHybridFeatures:
    """Tests for combined feature extraction."""

    @pytest.mark.unit
    def test_extract_hybrid_features(self, catalog_with_sources, simple_image):
        """Test hybrid (catalog + cutout) feature extraction."""
        features, names = rbf.extract_features(
            catalog_with_sources, simple_image, fwhm=3.0, method='hybrid'
        )

        # Should have both catalog and cutout features
        assert 'fwhm' in features or 'fwhm_ratio' in features
        assert 'sharpness' in features
        assert 'roundness' in features

    @pytest.mark.unit
    def test_auto_method_with_image(self, catalog_with_sources, simple_image):
        """Test auto method selects hybrid when image provided."""
        features, names = rbf.extract_features(
            catalog_with_sources, simple_image, method='auto'
        )

        # Should have cutout features
        assert 'sharpness' in features

    @pytest.mark.unit
    def test_auto_method_without_image(self, simple_catalog):
        """Test auto method selects catalog when no image."""
        features, names = rbf.extract_features(
            simple_catalog, image=None, method='auto'
        )

        # Should not have cutout features
        assert 'sharpness' not in features

    @pytest.mark.unit
    def test_invalid_method_raises(self, simple_catalog):
        """Test invalid extraction method validation."""
        with pytest.raises(ValueError, match="Unknown feature extraction method"):
            rbf.extract_features(simple_catalog, method='not_a_method')

    @pytest.mark.unit
    def test_empty_feature_set_raises(self):
        """Test clear error when no features can be extracted."""
        obj = Table()
        obj['x'] = [10.0, 20.0]
        obj['y'] = [10.0, 20.0]

        with pytest.raises(ValueError, match="No features could be extracted"):
            rbf.extract_features(obj, method='catalog')

    @pytest.mark.unit
    def test_feature_set_passthrough(self, simple_catalog):
        """extract_features should honor catalog feature_set selection."""
        default_features, _ = rbf.extract_features(
            simple_catalog, image=None, method='catalog', feature_set='default'
        )
        minimal_features, _ = rbf.extract_features(
            simple_catalog, image=None, method='catalog', feature_set='minimal'
        )

        assert 'snr' in default_features
        assert 'snr' not in minimal_features


# =============================================================================
# Tests: Trend Removal
# =============================================================================

class TestTrendRemoval:
    """Tests for spatial trend removal."""

    @pytest.mark.unit
    def test_trend_removal_basic(self, simple_catalog):
        """Test basic trend removal."""
        features, _ = rbf.extract_catalog_features(simple_catalog)

        detrended, models = rbf.remove_trends(
            features, simple_catalog, trend_cols=['x', 'y']
        )

        assert isinstance(detrended, dict)
        assert isinstance(models, dict)
        assert len(detrended) == len(features)

    @pytest.mark.unit
    def test_trend_removal_reduces_variance(self, simple_catalog):
        """Test that trend removal can reduce variance."""
        # Add artificial spatial trend
        features = {'test_feature': simple_catalog['x'] / 100 + np.random.normal(0, 0.1, len(simple_catalog))}

        detrended, _ = rbf.remove_trends(
            features, simple_catalog, trend_cols=['x', 'y']
        )

        # Detrended should have lower variance
        assert np.std(detrended['test_feature']) <= np.std(features['test_feature'])

    @pytest.mark.unit
    def test_apply_trend_models(self, simple_catalog):
        """Test applying trend models to new data."""
        features, _ = rbf.extract_catalog_features(simple_catalog)

        detrended, models = rbf.remove_trends(
            features, simple_catalog, trend_cols=['x', 'y']
        )

        # Apply to same data should give same result
        reapplied = rbf.apply_trend_models(features, simple_catalog, models, trend_cols=['x', 'y'])

        for name in detrended:
            if models.get(name) is not None:
                np.testing.assert_array_almost_equal(
                    detrended[name], reapplied[name], decimal=5
                )

    @pytest.mark.unit
    def test_apply_trend_models_missing_columns_raises(self, simple_catalog):
        """Trend model application should fail clearly on missing columns."""
        features, _ = rbf.extract_catalog_features(simple_catalog)
        _, models = rbf.remove_trends(features, simple_catalog, trend_cols=['x', 'y'])

        obj_missing = Table()
        obj_missing['x'] = simple_catalog['x']

        with pytest.raises(ValueError, match="Missing trend columns"):
            rbf.apply_trend_models(features, obj_missing, models, trend_cols=['x', 'y'])

    @pytest.mark.unit
    def test_apply_trend_models_reuses_fitted_columns(self, simple_catalog):
        """apply_trend_models should use the columns actually used during fitting."""
        features, _ = rbf.extract_catalog_features(simple_catalog)
        detrended, models = rbf.remove_trends(
            features, simple_catalog, trend_cols=['x', 'y', 'MAG_AUTO']
        )

        reapplied = rbf.apply_trend_models(
            features, simple_catalog, models, trend_cols=['x', 'y', 'MAG_AUTO']
        )

        for name in detrended:
            if models.get(name) is not None:
                np.testing.assert_array_almost_equal(
                    detrended[name], reapplied[name], decimal=5
                )

    @pytest.mark.unit
    def test_apply_trend_models_handles_nonfinite_positions(self, simple_catalog):
        """Rows with non-finite trend coordinates should not crash model application."""
        features, _ = rbf.extract_catalog_features(simple_catalog)
        _, models = rbf.remove_trends(features, simple_catalog, trend_cols=['x', 'y'])

        obj_bad = simple_catalog.copy()
        obj_bad['x'][0] = np.nan

        reapplied = rbf.apply_trend_models(features, obj_bad, models, trend_cols=['x', 'y'])

        for name in features:
            if np.isfinite(features[name][0]):
                assert reapplied[name][0] == pytest.approx(features[name][0])

    @pytest.mark.unit
    def test_trend_removal_uses_per_feature_validity(self, simple_catalog):
        """NaN-heavy features should not block detrending of clean features."""
        n = len(simple_catalog)
        rng = np.random.RandomState(0)

        features = {
            'clean_trend': np.array(simple_catalog['x']) / 10.0 + rng.normal(0, 0.01, n),
            'sparse': np.full(n, np.nan),
        }
        features['sparse'][:3] = [1.0, 2.0, 3.0]

        detrended, models = rbf.remove_trends(features, simple_catalog, trend_cols=['x', 'y'])

        assert models['clean_trend'] is not None
        assert np.nanstd(detrended['clean_trend']) < np.nanstd(features['clean_trend'])

    @pytest.mark.unit
    def test_remove_trends_no_available_columns_returns_copied_arrays(self, simple_catalog):
        """No-op detrending path should still deep-copy feature arrays."""
        features, _ = rbf.extract_catalog_features(simple_catalog)
        detrended, models = rbf.remove_trends(features, simple_catalog, trend_cols=['MISSING_COL'])

        assert isinstance(models, dict)
        for name in features:
            assert detrended[name] is not features[name]
            np.testing.assert_array_equal(detrended[name], features[name])

    @pytest.mark.unit
    def test_apply_trend_models_noop_returns_copied_arrays(self, simple_catalog):
        """Applying empty trend models should deep-copy feature arrays."""
        features, _ = rbf.extract_catalog_features(simple_catalog)
        detrended = rbf.apply_trend_models(features, simple_catalog, {})

        for name in features:
            assert detrended[name] is not features[name]
            np.testing.assert_array_equal(detrended[name], features[name])

    @pytest.mark.unit
    def test_trend_scales_align_with_available_columns(self, simple_catalog):
        """User-provided trend_scales should be re-aligned when columns are filtered."""
        features, _ = rbf.extract_catalog_features(simple_catalog)
        detrended, models = rbf.remove_trends(
            features,
            simple_catalog,
            trend_cols=['x', 'MISSING_COL', 'y'],
            trend_scales=[100.0, 50.0, 200.0],
        )

        assert models.trend_cols == ('x', 'y')
        assert isinstance(detrended, dict)


# =============================================================================
# Tests: Classifiers
# =============================================================================

class TestScoringClassifier:
    """Tests for rule-based scoring classifier."""

    @pytest.mark.unit
    def test_scoring_basic(self, catalog_with_sources, simple_image):
        """Test basic scoring classification."""
        features, _ = rbf.extract_features(
            catalog_with_sources, simple_image, fwhm=3.0, method='cutout'
        )

        clf = rbf.ScoringClassifier()
        scores = clf.predict_proba(features)

        assert len(scores) == len(catalog_with_sources)
        assert np.all((scores >= 0) & (scores <= 1))

    @pytest.mark.unit
    def test_scoring_predicts_artifacts_low(self, catalog_with_sources, simple_image):
        """Test that scoring gives artifacts lower scores."""
        features, _ = rbf.extract_features(
            catalog_with_sources, simple_image, fwhm=3.0, method='cutout'
        )

        clf = rbf.ScoringClassifier()
        scores = clf.predict_proba(features)

        # Real sources (first 5) should generally score higher than artifacts (last 2)
        real_mean = np.nanmean(scores[:5])
        artifact_mean = np.nanmean(scores[5:])

        # This is a soft test - artifacts should score lower on average
        assert artifact_mean <= real_mean


class TestIsolationForestClassifier:
    """Tests for IsolationForest classifier."""

    @pytest.mark.unit
    def test_isolation_basic(self, simple_catalog):
        """Test basic IsolationForest classification."""
        features, _ = rbf.extract_catalog_features(simple_catalog)

        clf = rbf.IsolationForestClassifier()
        clf.fit(features)

        predictions = clf.predict(features)
        assert len(predictions) == len(simple_catalog)
        assert set(predictions).issubset({-1, 1})

    @pytest.mark.unit
    def test_isolation_scores(self, simple_catalog):
        """Test IsolationForest score calculation."""
        features, _ = rbf.extract_catalog_features(simple_catalog)

        clf = rbf.IsolationForestClassifier()
        clf.fit(features)

        scores = clf.predict_proba(features)
        assert len(scores) == len(simple_catalog)
        assert np.all((scores >= 0) & (scores <= 1))


class TestRandomForestClassifier:
    """Tests for RandomForest classifier."""

    @pytest.mark.unit
    def test_randomforest_basic(self, simple_catalog):
        """Test basic RandomForest classification."""
        features, _ = rbf.extract_catalog_features(simple_catalog)
        labels = np.random.randint(0, 2, len(simple_catalog))

        clf = rbf.RandomForestClassifier(n_estimators=10)
        clf.fit(features, labels)

        predictions = clf.predict(features)
        assert len(predictions) == len(simple_catalog)

    @pytest.mark.unit
    def test_randomforest_feature_importances(self, simple_catalog):
        """Test RandomForest feature importances."""
        features, _ = rbf.extract_catalog_features(simple_catalog)
        labels = np.random.randint(0, 2, len(simple_catalog))

        clf = rbf.RandomForestClassifier(n_estimators=10)
        clf.fit(features, labels)

        importances = clf.feature_importances_
        assert importances is not None
        assert len(importances) == len(features)

    @pytest.mark.unit
    @pytest.mark.parametrize('label_value,expected', [(1, 1.0), (0, 0.0)])
    def test_randomforest_single_class_predict_proba(self, simple_catalog, label_value, expected):
        """Predict_proba should work even if fit data has a single class."""
        features, _ = rbf.extract_catalog_features(simple_catalog)
        labels = np.full(len(simple_catalog), label_value, dtype=int)

        clf = rbf.RandomForestClassifier(n_estimators=10)
        clf.fit(features, labels)

        scores = clf.predict_proba(features)
        assert np.allclose(scores, expected)


# =============================================================================
# Tests: High-Level API
# =============================================================================

class TestClassifyAPI:
    """Tests for high-level classify() function."""

    @pytest.mark.unit
    def test_classify_catalog_only(self, simple_catalog):
        """Test catalog-only classification."""
        result = rbf.classify(
            simple_catalog,
            method='catalog',
            classifier='isolation',
            add_score=True
        )

        assert 'rb_score' in result.colnames
        assert len(result) == len(simple_catalog)

    @pytest.mark.unit
    def test_classify_with_image(self, catalog_with_sources, simple_image):
        """Test classification with image."""
        result = rbf.classify(
            catalog_with_sources,
            simple_image,
            method='hybrid',
            classifier='scoring',
            add_score=True,
            flag_bogus=True
        )

        assert 'rb_score' in result.colnames
        # Check that some objects are flagged
        flagged = result['flags'] & 0x800
        assert np.any(flagged > 0) or np.all(result['rb_score'] >= 0.5)

    @pytest.mark.unit
    def test_classify_scoring_no_training(self, catalog_with_sources, simple_image):
        """Test that scoring classifier works without training."""
        # Should not raise
        result = rbf.classify(
            catalog_with_sources,
            simple_image,
            classifier='scoring',
            add_score=True
        )

        assert 'rb_score' in result.colnames

    @pytest.mark.unit
    def test_classify_with_trend_removal(self, simple_catalog):
        """Test classification with trend removal."""
        result = rbf.classify(
            simple_catalog,
            method='catalog',
            classifier='isolation',
            remove_trend=True,
            trend_cols=['x', 'y'],
            add_score=True
        )

        assert 'rb_score' in result.colnames

    @pytest.mark.unit
    def test_classify_uses_provided_isolation_model(self, simple_catalog):
        """classifier='isolation' should use provided model when passed."""
        class DummyModel:
            def predict_proba(self, features):
                n = len(next(iter(features.values())))
                return np.full(n, 0.123)

        result = rbf.classify(
            simple_catalog,
            method='catalog',
            classifier='isolation',
            model=DummyModel(),
            remove_trend=False,
            add_score=True,
            flag_bogus=False
        )

        assert np.allclose(result['rb_score'], 0.123)

    @pytest.mark.unit
    def test_classify_clears_previous_bogus_flag(self, simple_catalog):
        """Reclassification should overwrite stale 0x800 bogus flags."""
        obj = simple_catalog.copy()
        obj['flags'] |= 0x800

        result = rbf.classify(
            obj,
            method='catalog',
            classifier='scoring',
            threshold=0.0,
            remove_trend=False,
            add_score=True,
            flag_bogus=True
        )

        assert np.all((result['flags'] & 0x800) == 0)


class TestTraining:
    """Tests for classifier training."""

    @pytest.mark.unit
    def test_train_classifier_basic(self, simple_catalog):
        """Test basic classifier training."""
        features, _ = rbf.extract_catalog_features(simple_catalog)
        labels = np.random.randint(0, 2, len(simple_catalog))

        clf, metrics = rbf.train_classifier(
            features, labels,
            classifier='randomforest',
            test_size=0.3
        )

        assert clf is not None
        assert 'test_accuracy' in metrics
        assert 0 <= metrics['test_accuracy'] <= 1

    @pytest.mark.unit
    def test_generate_training_data_zero_images(self):
        """Generating zero images should return empty outputs without crashing."""
        features, labels = rbf.generate_training_data(n_images=0)
        assert features == {}
        assert isinstance(labels, np.ndarray)
        assert labels.size == 0

    @pytest.mark.unit
    def test_train_classifier_rejects_nonfinite_rows(self):
        """train_classifier should reject datasets with no finite feature rows."""
        features = {
            'f1': np.array([np.nan, np.nan, np.nan, np.nan]),
            'f2': np.array([np.nan, np.nan, np.nan, np.nan]),
        }
        labels = np.array([0, 1, 0, 1])

        with pytest.raises(ValueError, match="No finite feature rows available for training"):
            rbf.train_classifier(features, labels, classifier='randomforest', test_size=0.5)

    @pytest.mark.unit
    def test_train_classifier_requires_two_classes(self):
        """Stratified training should reject single-class labels."""
        features = {
            'f1': np.array([1.0, 2.0, 3.0, 4.0]),
            'f2': np.array([2.0, 3.0, 4.0, 5.0]),
        }
        labels = np.zeros(4, dtype=int)

        with pytest.raises(ValueError, match="at least two label classes"):
            rbf.train_classifier(features, labels, classifier='randomforest', test_size=0.5)

    @pytest.mark.unit
    def test_train_classifier_requires_two_samples_per_class(self):
        """Stratified training should reject classes with only one sample."""
        features = {
            'f1': np.array([1.0, 2.0, 3.0, 4.0]),
            'f2': np.array([2.0, 3.0, 4.0, 5.0]),
        }
        labels = np.array([0, 0, 0, 1], dtype=int)

        with pytest.raises(ValueError, match="Each class needs at least 2 samples"):
            rbf.train_classifier(features, labels, classifier='randomforest', test_size=0.5)


class TestFeatureMathEdgeCases:
    """Tests for feature computation edge cases."""

    @pytest.mark.unit
    def test_roundness_with_negative_background(self):
        """Roundness should still be finite with positive core on negative background."""
        cutout = np.full((5, 5), -1.0)
        cutout[2, 2] = 5.0
        geom = rbf._CutoutGeometry(cutout.shape)
        roundness = rbf._compute_roundness(cutout, geom)

        assert np.isfinite(roundness)
        assert 0 <= roundness <= 1
