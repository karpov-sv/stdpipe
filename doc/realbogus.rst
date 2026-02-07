Real-Bogus Classification
==========================

*STDPipe* provides two complementary approaches for classifying detected objects as real astronomical sources (stars, galaxies) or artifacts (cosmic rays, hot pixels, satellite trails, etc.):

1. **Feature-based classification** (:mod:`stdpipe.realbogus_features`) - uses explicit morphological features with sklearn classifiers
2. **CNN-based classification** (:mod:`stdpipe.realbogus`) - uses deep learning for maximum accuracy

This page documents both approaches and helps you choose the right one for your use case.

Quick Start
-----------

**Feature-based (no TensorFlow needed):**

.. code-block:: python

   from stdpipe import photometry, realbogus_features as rbf

   # Detect objects
   obj = photometry.get_objects_sep(image, thresh=3.0)

   # Classify with scoring (no training needed)
   obj = rbf.classify(obj, image, classifier='scoring',
                      threshold=0.5, add_score=True, flag_bogus=True)

   # Filter to real objects
   real = obj[obj['rb_score'] >= 0.5]


**CNN-based (requires TensorFlow):**

.. code-block:: python

   from stdpipe import photometry, realbogus

   # Detect objects
   obj = photometry.get_objects_sep(image, thresh=3.0)

   # Classify with CNN
   obj = realbogus.classify_realbogus(obj, image, threshold=0.5,
                                       add_score=True, flag_bogus=True)

   # Filter to real objects
   real = obj[obj['rb_score'] >= 0.5]


Comparison of Approaches
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Feature-based
     - CNN-based
   * - **Dependencies**
     - sklearn only (already required)
     - TensorFlow required
   * - **Training data**
     - 100s of examples
     - 1000s of examples
   * - **Training time**
     - Seconds
     - Minutes to hours
   * - **Inference speed**
     - ~1ms per object
     - ~10ms per object
   * - **GPU required**
     - No
     - Optional but helpful
   * - **Interpretability**
     - High (explicit features)
     - Low (black box)
   * - **Accuracy**
     - Good for typical artifacts
     - Better for complex cases
   * - **Customization**
     - Easy (adjust weights/thresholds)
     - Requires retraining

**Recommendation:**

- Use feature-based for quick filtering and when interpretability matters
- Use CNN-based when maximum accuracy is needed and TensorFlow is available
- Can use both: feature-based for quick pre-filtering, CNN for detailed classification


Feature-Based Classification
-----------------------------

The :mod:`stdpipe.realbogus_features` module provides real-bogus classification using explicit morphological features extracted from catalogs and image cutouts.

Key advantages:

- **No TensorFlow dependency** - works on any system with sklearn
- **Interpretable** - can explain why an object is classified as bogus
- **Fast** - no GPU required, milliseconds per object
- **Small training data** - can train on hundreds of examples
- **Customizable** - features and thresholds can be tuned per instrument
- **Multiple modes** - catalog-only, cutout-based, or hybrid


Feature Extraction
^^^^^^^^^^^^^^^^^^

Features are extracted from two sources:

**Catalog Features** (from detection catalog, no image needed):

- ``fwhm``, ``fwhm_ratio`` - PSF size and consistency
- ``ellipticity``, ``elongation`` - Shape parameters (high for trails)
- ``peakiness`` - FLUX_MAX / FLUX_AUTO (high for cosmic rays)
- ``snr`` - Signal-to-noise ratio

**Cutout Features** (from image cutouts):

- ``sharpness`` - Central concentration (high = cosmic ray/hot pixel)
- ``concentration`` - Flux distribution ratio
- ``symmetry`` - Rotational symmetry (high = asymmetric artifact)
- ``roundness`` - Shape roundness (low = trail/elongated)
- ``psf_match`` - PSF fit quality χ²
- ``peak_offset`` - Centroid-to-peak distance
- ``edge_gradient`` - Edge sharpness (high = sharp cosmic ray edge)
- ``bg_consistency`` - Background uniformity in annulus

You can use catalog-only features when you don't have the image, cutout-only for maximum accuracy, or hybrid mode combining both.


Classifiers
^^^^^^^^^^^

Three classifier types are available:

**1. Scoring Classifier (No Training)**

Rule-based scoring using predefined weights and thresholds. Good for quick filtering without any training data.

.. code-block:: python

   # Use scoring classifier (no training needed)
   obj = rbf.classify(obj, image, classifier='scoring',
                      threshold=0.5, add_score=True)

   # Examine scores
   print(f"Mean score: {obj['rb_score'].mean():.2f}")
   print(f"Real sources: {sum(obj['rb_score'] > 0.5)}")

**2. Isolation Forest (Unsupervised)**

Anomaly detection that learns "normal" from the data. Good when you have mostly real sources and want to find outliers.

.. code-block:: python

   # Classify with IsolationForest (learns from data)
   obj = rbf.classify(obj, image, classifier='isolation',
                      remove_trend=True, add_score=True)

   # Objects with low scores are outliers (likely artifacts)
   artifacts = obj[obj['rb_score'] < 0.3]

**3. Random Forest (Supervised)**

Traditional supervised classification. Requires labeled training data but provides best accuracy.

.. code-block:: python

   # Train on labeled data
   features, names = rbf.extract_features(train_obj, train_image, method='hybrid')
   clf, metrics = rbf.train_classifier(features, train_labels,
                                       classifier='randomforest')

   print(f"Accuracy: {metrics['test_accuracy']:.1%}")

   # Apply to new data
   obj = rbf.classify(obj, image, classifier='randomforest',
                      model=clf, add_score=True)


Trend Removal
^^^^^^^^^^^^^

Many features vary systematically across the image (due to PSF variation, vignetting) or with magnitude. Trend removal normalizes features to make classification more robust:

.. code-block:: python

   # Enable spatial trend removal
   obj = rbf.classify(obj, image,
                      remove_trend=True,
                      trend_cols=['x', 'y'],  # Spatial trends
                      add_score=True)

   # Can also include magnitude trends
   obj = rbf.classify(obj, image,
                      remove_trend=True,
                      trend_cols=['x', 'y', 'MAG_AUTO'],
                      add_score=True)


Training and Evaluation Tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``examples/train_realbogus_features.py`` script provides a complete command-line tool for training, testing, and evaluating classifiers:

**Train a classifier:**

.. code-block:: bash

   # Train RandomForest on simulated data
   python train_realbogus_features.py train --n-images 100 --output model.pkl

   # Train with hybrid features and trend removal
   python train_realbogus_features.py train \
       --method hybrid --trend-removal --n-images 200 --output model.pkl

**Test on single image:**

.. code-block:: bash

   # Test scoring classifier (no training needed)
   python train_realbogus_features.py test --classifier scoring

   # Test trained model
   python train_realbogus_features.py test --model model.pkl

   # Test on real image
   python train_realbogus_features.py test --model model.pkl \
       --image myimage.fits --fwhm 3.5

**Evaluate on multiple images:**

.. code-block:: bash

   # Comprehensive evaluation
   python train_realbogus_features.py evaluate \
       --model model.pkl --n-images 50 --output results/

**Compare feature methods:**

.. code-block:: bash

   # Compare catalog/cutout/hybrid methods
   python train_realbogus_features.py compare --output comparison.png

The tool automatically generates comprehensive visualizations and performance metrics.


Customization
^^^^^^^^^^^^^

The scoring classifier can be customized for your instrument:

.. code-block:: python

   # Define custom rules
   custom_rules = {
       'sharpness': {'weight': 0.2, 'ideal': 1.3, 'bad': 'high', 'threshold': 4.0},
       'roundness': {'weight': 0.3, 'ideal': 1.0, 'bad': 'low', 'threshold': 0.2},
       'fwhm_ratio': {'weight': 0.2, 'ideal': 1.0, 'bad': 'both', 'threshold': 0.3},
   }

   clf = rbf.ScoringClassifier(rules=custom_rules)
   features, _ = rbf.extract_features(obj, image, method='hybrid')
   scores = clf.predict_proba(features)


Integration with artefacts.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The legacy :mod:`stdpipe.artefacts` module now uses :mod:`stdpipe.realbogus_features` internally:

.. code-block:: python

   from stdpipe import artefacts

   # New unified function
   good = artefacts.filter_detections(
       obj, image,
       method='hybrid',
       classifier='isolation',
       remove_trend=True
   )


API Reference
^^^^^^^^^^^^^

.. autofunction:: stdpipe.realbogus_features.classify
   :noindex:

.. autofunction:: stdpipe.realbogus_features.extract_features
   :noindex:

.. autofunction:: stdpipe.realbogus_features.train_classifier
   :noindex:

.. autoclass:: stdpipe.realbogus_features.ScoringClassifier
   :members:
   :noindex:

.. autoclass:: stdpipe.realbogus_features.IsolationForestClassifier
   :members:
   :noindex:

.. autoclass:: stdpipe.realbogus_features.RandomForestClassifier
   :members:
   :noindex:


CNN-Based Classification
------------------------

The :mod:`stdpipe.realbogus` module provides deep learning-based classification using convolutional neural networks (CNNs). This approach learns features automatically from training data and can achieve higher accuracy than feature-based methods, especially for complex artifact types.

Requirements
^^^^^^^^^^^^

CNN-based classification requires TensorFlow:

.. code-block:: bash

   pip install tensorflow

   # Or with GPU support
   pip install tensorflow[and-cuda]


Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from stdpipe import realbogus

   # Classify with pre-trained model (if available)
   obj = realbogus.classify_realbogus(
       obj, image,
       threshold=0.5,
       add_score=True,
       flag_bogus=True
   )

   # Filter to real objects
   real = obj[obj['rb_score'] >= 0.5]
   print(f"Kept {len(real)}/{len(obj)} objects")


Training Custom Models
^^^^^^^^^^^^^^^^^^^^^^

For best results on your specific instrument, train a custom CNN model:

.. code-block:: python

   from stdpipe import realbogus

   # Prepare training data (cutouts and labels)
   # labels: 1 = real, 0 = bogus
   train_cutouts = realbogus.extract_cutouts(train_obj, train_image)

   # Train CNN model
   model = realbogus.train_model(
       train_cutouts, train_labels,
       epochs=50,
       batch_size=32,
       validation_split=0.2
   )

   # Save model
   model.save('my_realbogus_model.h5')

   # Apply to new data
   obj = realbogus.classify_realbogus(
       obj, image,
       model='my_realbogus_model.h5',
       threshold=0.5
   )


API Reference
^^^^^^^^^^^^^

.. autofunction:: stdpipe.realbogus.classify_realbogus
   :noindex:


Combining Both Approaches
--------------------------

For optimal performance, you can use both approaches in a pipeline:

.. code-block:: python

   from stdpipe import photometry, realbogus_features as rbf

   # Detect objects
   obj = photometry.get_objects_sep(image, thresh=3.0)

   # Quick pre-filter with feature-based classifier
   obj = rbf.classify(obj, image, classifier='scoring', threshold=0.3)
   candidates = obj[obj['rb_score'] >= 0.3]

   print(f"Pre-filtered to {len(candidates)} candidates")

   # Detailed classification with CNN (if available)
   try:
       from stdpipe import realbogus
       candidates = realbogus.classify_realbogus(
           candidates, image,
           threshold=0.5,
           add_score=True
       )
       # Override rb_score with CNN score
   except ImportError:
       print("TensorFlow not available, using feature-based scores")

   # Final filtering
   real = candidates[candidates['rb_score'] >= 0.5]

This two-stage approach:

1. Quickly filters obvious artifacts with feature-based classifier
2. Applies more computationally expensive CNN only to candidates
3. Reduces overall computation time while maintaining high accuracy


Output Flags
------------

Both classification methods use the same flag convention:

.. code-block:: python

   # Flag 0x800 marks objects classified as bogus
   bogus_mask = (obj['flags'] & 0x800) != 0
   real_mask = (obj['flags'] & 0x800) == 0

   # Clear bogus flags if needed
   obj['flags'] &= ~0x800

Both methods also add a ``rb_score`` column with values in [0, 1], where:

- Values near 1.0 indicate high confidence the object is real
- Values near 0.0 indicate high confidence the object is bogus
- Typical threshold is 0.5, but can be adjusted based on your requirements


Performance Considerations
--------------------------

**Feature-based:**

- **Speed**: 1000-5000 objects/second (hybrid mode with scoring)
- **Memory**: ~1 KB per object for cutouts, ~100 bytes for features
- **Optimization**: Use catalog-only mode when image features aren't needed

**CNN-based:**

- **Speed**: 100-500 objects/second (CPU), 1000-5000 objects/second (GPU)
- **Memory**: Larger cutouts (~4 KB per object), model weights (~10-100 MB)
- **Optimization**: Batch process objects, use GPU if available


Troubleshooting
---------------

**All objects classified as bogus:**

- Check that FWHM is correctly estimated
- Try lowering the threshold (e.g., 0.3 instead of 0.5)
- Verify image units are linear (not log scale)
- For feature-based: examine feature distributions to tune rules

**Cosmic rays not detected:**

- For feature-based: increase ``sharpness`` weight in scoring rules
- Ensure cutout radius is large enough to capture surrounding background
- Check that background subtraction is working correctly

**Bright stars classified as bogus:**

- Saturated stars have unusual profiles and may be misclassified
- Add saturation mask before classification
- Use ``flags`` column to exclude saturated objects before classification

**IsolationForest marks rare objects as outliers:**

- This is expected (outlier = unusual)
- Use supervised RandomForest if you have labeled data
- Or use scoring classifier which doesn't learn from data


References
----------

**Feature-based classification:**

- Bloom, J. S., et al. (2012). "Automating Discovery and Classification of Transients and Variable Stars in the Synoptic Survey Era." PASP 124:1175
- Wright, D. E., et al. (2015). "A Machine Learning Approach for Dynamical Mass Measurements of Galaxy Clusters." ApJ 809:159
- Liu, F. T., et al. (2008). "Isolation Forest." ICDM 2008

**CNN-based classification:**

- Duev, D. A., et al. (2019). "Real-bogus classification for the Zwicky Transient Facility using deep learning." MNRAS 489:3582
- Cabrera-Vives, G., et al. (2017). "Deep-HiTS: Rotation Invariant Convolutional Neural Network for Transient Detection." ApJ 836:97


Example Workflows
-----------------

**Workflow 1: Quick Filtering for Survey Data**

.. code-block:: python

   from stdpipe import photometry, realbogus_features as rbf

   # Detect objects
   obj = photometry.get_objects_sep(image, thresh=3.0, aper=5.0)
   print(f"Detected {len(obj)} objects")

   # Quick filtering with scoring classifier
   obj = rbf.classify(obj, image, classifier='scoring',
                      method='hybrid', threshold=0.5,
                      add_score=True, flag_bogus=True)

   # Filter to real sources
   real = obj[(obj['rb_score'] >= 0.5) & ((obj['flags'] & 0x800) == 0)]
   print(f"Kept {len(real)} real sources")


**Workflow 2: High-Precision Classification**

.. code-block:: python

   from stdpipe import photometry, realbogus_features as rbf

   # Detect objects
   obj = photometry.get_objects_sep(image, thresh=2.5)

   # Train IsolationForest on this specific image
   features, names = rbf.extract_features(obj, image, method='hybrid')
   clf = rbf.IsolationForestClassifier(contamination=0.1)
   clf.fit(features)

   # Classify with trend removal
   obj = rbf.classify(obj, image, classifier='isolation',
                      model=clf, remove_trend=True,
                      trend_cols=['x', 'y'], add_score=True)

   # Conservative threshold
   real = obj[obj['rb_score'] >= 0.7]


**Workflow 3: Building a Labeled Training Set**

.. code-block:: python

   from stdpipe import photometry, realbogus_features as rbf
   import numpy as np

   # Use scoring classifier to get initial classifications
   obj = rbf.classify(obj, image, classifier='scoring',
                      method='hybrid', add_score=True)

   # High-confidence real sources
   certain_real = obj[obj['rb_score'] > 0.9]

   # High-confidence bogus sources
   certain_bogus = obj[obj['rb_score'] < 0.1]

   # Uncertain objects for manual review
   uncertain = obj[(obj['rb_score'] >= 0.1) & (obj['rb_score'] <= 0.9)]
   print(f"Need manual review for {len(uncertain)} objects")

   # Build training labels
   labels = np.concatenate([
       np.ones(len(certain_real)),
       np.zeros(len(certain_bogus)),
       manual_labels  # From visual inspection
   ])

   # Train RandomForest
   all_obj = Table(rows=list(certain_real) + list(certain_bogus) + list(uncertain))
   features, _ = rbf.extract_features(all_obj, image, method='hybrid')
   clf, metrics = rbf.train_classifier(features, labels,
                                       classifier='randomforest')

   print(f"Trained model accuracy: {metrics['test_accuracy']:.1%}")


**Workflow 4: Cross-Validation on Multiple Images**

.. code-block:: python

   from stdpipe import realbogus_features as rbf
   from sklearn.model_selection import cross_val_score
   import numpy as np

   # Collect features from multiple images
   all_features = []
   all_labels = []

   for image, truth_catalog in zip(images, truth_catalogs):
       obj = photometry.get_objects_sep(image, thresh=3.0)
       features, _ = rbf.extract_features(obj, image, method='hybrid')

       # Match to truth catalog for labels
       labels = match_to_truth(obj, truth_catalog)

       all_features.append(features)
       all_labels.extend(labels)

   # Combine features
   combined_features = {
       key: np.concatenate([f[key] for f in all_features])
       for key in all_features[0].keys()
   }

   # Cross-validation
   clf = rbf.RandomForestClassifier(n_estimators=100)
   scores = cross_val_score(clf, combined_features, all_labels, cv=5)

   print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
