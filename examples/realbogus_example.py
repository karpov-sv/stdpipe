#!/usr/bin/env python3
"""
Real-Bogus Classifier Example

This script demonstrates the complete workflow for using STDPipe's real-bogus
classifier to filter artifacts from astronomical object detections.

The workflow includes:
1. Training a CNN classifier on simulated data
2. Applying the classifier to filter detections
3. Comparing before/after classification
4. Saving and loading trained models

Requirements:
    - STDPipe with ML dependencies: pip install stdpipe[ml]
    - Or just: pip install stdpipe tensorflow

Author: STDPipe Contributors
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# Check if real-bogus classifier is available
try:
    from stdpipe import realbogus, simulation, photometry
    if not realbogus.HAS_TENSORFLOW:
        raise ImportError("TensorFlow not found")
except ImportError as e:
    print("Error: Real-bogus classifier requires TensorFlow")
    print("Install with: pip install stdpipe[ml]")
    print(f"Details: {e}")
    exit(1)


def example_1_train_classifier():
    """
    Example 1: Train a real-bogus classifier on simulated data.

    This creates a CNN model and trains it on simulated images containing
    both real sources (stars, galaxies) and artifacts (cosmic rays, hot pixels).
    """
    print("=" * 70)
    print("Example 1: Training Real-Bogus Classifier")
    print("=" * 70)

    # Generate training data from simulated images
    print("\nGenerating training data from simulated images...")
    print("(This may take a few minutes for realistic dataset sizes)")

    # For this example, use small dataset (fast training)
    # For production, increase n_simulated to 500-1000
    model, history = realbogus.train_realbogus_classifier(
        n_simulated=20,  # Number of simulated images (increase for better model)
        image_size=(512, 512),  # Size of simulated images
        fwhm_range=(2.0, 6.0),  # Range of FWHM values (handles variable seeing)
        epochs=10,  # Training epochs (increase for better convergence)
        batch_size=64,
        model_file='realbogus_model.h5',  # Save trained model
        verbose=True
    )

    print("\nTraining complete!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.3f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
    print("Model saved to: realbogus_model.h5")

    return model


def example_2_classify_detections(model=None):
    """
    Example 2: Apply real-bogus classifier to filter detections.

    This simulates an image with both real sources and artifacts,
    detects objects, and classifies them as real or bogus.
    """
    print("\n" + "=" * 70)
    print("Example 2: Classifying Detections")
    print("=" * 70)

    # Load model if not provided
    if model is None:
        print("\nLoading pre-trained model...")
        try:
            model = realbogus.load_realbogus_model(model_file='realbogus_model.h5')
            print("Model loaded successfully")
        except FileNotFoundError:
            print("Error: No trained model found. Run example_1_train_classifier() first.")
            return

    # Simulate a realistic astronomical image
    print("\nSimulating test image...")
    sim = simulation.simulate_image(
        width=1024,
        height=1024,
        n_stars=50,
        star_fwhm=3.5,
        n_galaxies=15,
        n_cosmic_rays=10,
        n_hot_pixels=20,
        n_satellites=1,
        background=1000.0,
        verbose=True
    )

    image = sim['image']
    truth_catalog = sim['catalog']

    print(f"\nInjected sources:")
    print(f"  Real sources: {len(truth_catalog[truth_catalog['type'] == 'star'])} stars")
    print(f"  Real sources: {len(truth_catalog[truth_catalog['type'] == 'galaxy'])} galaxies")
    print(f"  Artifacts: {len(truth_catalog[truth_catalog['type'] != 'star']) - len(truth_catalog[truth_catalog['type'] == 'galaxy'])} total")

    # Detect objects (this includes both real sources and artifacts)
    print("\nDetecting objects...")
    obj_detected = photometry.get_objects_sep(
        image,
        thresh=3.0,
        aper=8.0,
        verbose=False
    )

    print(f"Detected {len(obj_detected)} objects (real + artifacts)")

    # Classify detections as real or bogus
    print("\nClassifying detections with CNN...")
    obj_clean = realbogus.classify_realbogus(
        obj_detected,
        image,
        model=model,
        threshold=0.5,  # Objects with score > 0.5 are real
        add_score=True,  # Add 'rb_score' column
        flag_bogus=True,  # Filter out bogus detections
        verbose=True
    )

    print(f"\nAfter filtering: {len(obj_clean)} real sources retained")
    print(f"Rejected: {len(obj_detected) - len(obj_clean)} artifacts")
    print(f"Rejection rate: {100 * (1 - len(obj_clean)/len(obj_detected)):.1f}%")

    # Show distribution of real-bogus scores
    print("\nReal-bogus score distribution:")
    scores = obj_detected['rb_score'][~np.isnan(obj_detected['rb_score'])]
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Median: {np.median(scores):.3f}")
    print(f"  Min: {np.min(scores):.3f}, Max: {np.max(scores):.3f}")

    return {
        'image': image,
        'truth': truth_catalog,
        'detected': obj_detected,
        'clean': obj_clean
    }


def example_3_score_without_filtering():
    """
    Example 3: Add real-bogus scores without filtering.

    Sometimes you want to keep all detections but add classification
    scores for downstream analysis or manual inspection.
    """
    print("\n" + "=" * 70)
    print("Example 3: Adding Scores Without Filtering")
    print("=" * 70)

    # Load model
    try:
        model = realbogus.load_realbogus_model(model_file='realbogus_model.h5')
    except FileNotFoundError:
        print("Error: No trained model found. Run example_1_train_classifier() first.")
        return

    # Simulate and detect
    print("\nSimulating image and detecting objects...")
    sim = simulation.simulate_image(
        width=512, height=512,
        n_stars=30,
        star_fwhm=3.0,
        n_cosmic_rays=8,
        verbose=False
    )

    obj = photometry.get_objects_sep(sim['image'], thresh=3.0, verbose=False)

    # Add scores without filtering
    print(f"\nAdding real-bogus scores to {len(obj)} detections...")
    obj_scored = realbogus.classify_realbogus(
        obj,
        sim['image'],
        model=model,
        add_score=True,
        flag_bogus=False,  # Don't filter, just add scores
        verbose=False
    )

    print("\nAll detections retained with scores:")
    print(obj_scored[['x', 'y', 'flux', 'rb_score']])

    # You can now filter manually based on score
    high_confidence = obj_scored[obj_scored['rb_score'] > 0.8]
    print(f"\nHigh-confidence detections (score > 0.8): {len(high_confidence)}")

    return obj_scored


def example_4_different_thresholds():
    """
    Example 4: Compare different classification thresholds.

    The threshold controls how strict the classifier is. Lower threshold
    accepts more objects (higher completeness), higher threshold rejects
    more artifacts (higher purity).
    """
    print("\n" + "=" * 70)
    print("Example 4: Comparing Classification Thresholds")
    print("=" * 70)

    # Load model
    try:
        model = realbogus.load_realbogus_model(model_file='realbogus_model.h5')
    except FileNotFoundError:
        print("Error: No trained model found. Run example_1_train_classifier() first.")
        return

    # Simulate and detect
    print("\nSimulating image...")
    sim = simulation.simulate_image(
        width=1024, height=1024,
        n_stars=40,
        star_fwhm=3.5,
        n_cosmic_rays=15,
        n_hot_pixels=25,
        verbose=False
    )

    obj = photometry.get_objects_sep(sim['image'], thresh=3.0, verbose=False)

    print(f"Total detections: {len(obj)}")

    # Try different thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]

    print("\nThreshold comparison:")
    print("  Threshold | Retained | Rejected | Rejection %")
    print("  ----------|----------|----------|------------")

    for thresh in thresholds:
        obj_filtered = realbogus.classify_realbogus(
            obj.copy(),
            sim['image'],
            model=model,
            threshold=thresh,
            flag_bogus=True,
            verbose=False
        )

        n_retained = len(obj_filtered)
        n_rejected = len(obj) - n_retained
        rejection_pct = 100 * n_rejected / len(obj)

        print(f"  {thresh:8.1f} | {n_retained:8d} | {n_rejected:8d} | {rejection_pct:10.1f}%")

    print("\nRecommendation:")
    print("  - Use threshold=0.3-0.5 for high completeness (keep more real sources)")
    print("  - Use threshold=0.5-0.7 for balanced performance (default)")
    print("  - Use threshold=0.7-0.9 for high purity (reject more artifacts)")


def example_5_custom_training_data():
    """
    Example 5: Train with custom parameters.

    This shows how to customize the training data generation
    for specific observing conditions (FWHM range, artifact rates, etc.)
    """
    print("\n" + "=" * 70)
    print("Example 5: Training with Custom Parameters")
    print("=" * 70)

    # Generate training data with custom parameters
    print("\nGenerating custom training data...")
    print("Simulating conditions: poor seeing, high artifact rate")

    data = simulation.generate_realbogus_training_data(
        n_images=10,
        image_size=(512, 512),
        n_stars_range=(20, 50),
        n_galaxies_range=(5, 15),
        fwhm_range=(4.0, 10.0),  # Poor seeing conditions
        n_cosmic_rays_range=(10, 30),  # High cosmic ray rate
        n_hot_pixels_range=(15, 40),  # Many hot pixels
        augment=True,
        verbose=True
    )

    print(f"\nGenerated {len(data['X'])} training samples")
    print(f"  Real sources: {np.sum(data['y'])}")
    print(f"  Artifacts: {np.sum(~data['y'].astype(bool))}")

    # Train model on custom data
    print("\nTraining model on custom data...")
    model, history = realbogus.train_realbogus_classifier(
        training_data=data,
        epochs=10,
        model_file='realbogus_custom_model.h5',
        verbose=True
    )

    print("\nCustom model saved to: realbogus_custom_model.h5")
    print("This model is optimized for poor seeing and high artifact rates")

    return model


def visualize_results(results):
    """
    Visualize classification results.

    Creates a figure showing:
    - Original detections
    - Filtered detections
    - Real-bogus score distribution
    """
    if results is None:
        print("No results to visualize")
        return

    image = results['image']
    detected = results['detected']
    clean = results['clean']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image with all detections
    ax = axes[0]
    ax.imshow(image, origin='lower', cmap='gray', vmin=900, vmax=1200)
    ax.scatter(detected['x'], detected['y'], s=50, facecolors='none',
               edgecolors='red', linewidth=1, alpha=0.7)
    ax.set_title(f'All Detections ({len(detected)})')
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')

    # Image with filtered (real) detections
    ax = axes[1]
    ax.imshow(image, origin='lower', cmap='gray', vmin=900, vmax=1200)
    ax.scatter(clean['x'], clean['y'], s=50, facecolors='none',
               edgecolors='green', linewidth=1, alpha=0.7)
    ax.set_title(f'Real Sources ({len(clean)})')
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')

    # Real-bogus score histogram
    ax = axes[2]
    scores = detected['rb_score'][~np.isnan(detected['rb_score'])]
    ax.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Real-Bogus Score')
    ax.set_ylabel('Number of Detections')
    ax.set_title('Score Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig('realbogus_results.png', dpi=150)
    print("\nVisualization saved to: realbogus_results.png")
    plt.show()


def main():
    """
    Run all examples in sequence.
    """
    print("\n" + "=" * 70)
    print("STDPipe Real-Bogus Classifier Examples")
    print("=" * 70)
    print("\nThis script demonstrates the complete workflow for training")
    print("and using a CNN-based real-bogus classifier.\n")

    # Example 1: Train classifier
    print("Running Example 1: Training...")
    model = example_1_train_classifier()

    # Example 2: Classify detections
    print("\n\nRunning Example 2: Classification...")
    results = example_2_classify_detections(model)

    # Example 3: Add scores without filtering
    print("\n\nRunning Example 3: Scoring...")
    example_3_score_without_filtering()

    # Example 4: Compare thresholds
    print("\n\nRunning Example 4: Threshold comparison...")
    example_4_different_thresholds()

    # Example 5: Custom training
    print("\n\nRunning Example 5: Custom training...")
    example_5_custom_training_data()

    # Visualize results
    print("\n\nGenerating visualization...")
    visualize_results(results)

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Try the classifier on your own astronomical images")
    print("  2. Adjust the threshold based on your science case")
    print("  3. Fine-tune the model with real labeled data if available")
    print("  4. Integrate into your pipeline with classify_realbogus()")


if __name__ == '__main__':
    main()
