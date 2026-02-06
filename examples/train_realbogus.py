#!/usr/bin/env python3
"""
Real-Bogus Classifier Training and Evaluation Tool

A command-line tool for training, testing, and evaluating CNN-based real-bogus
classifiers for astronomical object detection.

Usage Examples:
    # Train a new classifier (stars-only by default)
    python train_realbogus.py train --n-images 500 --epochs 30 --output model.h5

    # Train classifier that includes galaxies as real sources
    python train_realbogus.py train --include-galaxies --n-images 500 --output full_model.h5

    # Test on single image with optical aberrations (interactive)
    python train_realbogus.py test --model model.h5 --enable-aberrations --interactive

    # Evaluate with aberrated PSFs (30% of images)
    python train_realbogus.py evaluate --model model.h5 --n-images 50 --enable-aberrations \\
        --aberration-fraction 0.3 --output results/

    # Evaluate with strong aberrations
    python train_realbogus.py evaluate --model model.h5 --enable-aberrations \\
        --defocus-max 3.0 --astigmatism-max 2.0 --coma-max 1.5 --output results_aberrated/

Author: STDPipe Contributors
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend by default
import matplotlib.pyplot as plt
from astropy.table import Table

# Check dependencies
try:
    from stdpipe import realbogus, simulation, photometry
    if not realbogus.HAS_TENSORFLOW:
        raise ImportError("TensorFlow not found")
except ImportError as e:
    print(f"Error: Missing dependencies - {e}")
    print("Install with: pip install stdpipe[ml]")
    sys.exit(1)


def create_aberrated_psf_model(fwhm, enable_aberrations=False, aberration_params=None):
    """
    Create a PSF model with optional random optical aberrations.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum in pixels
    enable_aberrations : bool
        Whether to add optical aberrations
    aberration_params : dict, optional
        Dictionary with aberration ranges:
        - defocus_range: (min, max) in waves
        - astigmatism_range: (min, max) in waves
        - coma_range: (min, max) in waves
        - fraction: fraction of PSFs that have aberrations (default 1.0)

    Returns
    -------
    psf_model : dict
        PSFEx-compatible PSF model
    """
    if not enable_aberrations or aberration_params is None:
        # Return standard Gaussian PSF
        return simulation.create_psf_model(fwhm=fwhm, psf_type='gaussian')

    # Check if this PSF should have aberrations
    fraction = aberration_params.get('fraction', 1.0)
    if np.random.random() > fraction:
        # Return standard PSF (no aberrations)
        return simulation.create_psf_model(fwhm=fwhm, psf_type='gaussian')

    # Random aberrations within specified ranges
    defocus_range = aberration_params.get('defocus_range', (0.0, 0.0))
    astigmatism_range = aberration_params.get('astigmatism_range', (0.0, 0.0))
    coma_range = aberration_params.get('coma_range', (0.0, 0.0))

    # Random aberration coefficients
    defocus = np.random.uniform(*defocus_range) if defocus_range[1] > 0 else 0.0
    astigmatism_x = np.random.uniform(*astigmatism_range) if astigmatism_range[1] > 0 else 0.0
    astigmatism_y = np.random.uniform(*astigmatism_range) if astigmatism_range[1] > 0 else 0.0
    coma_x = np.random.uniform(*coma_range) if coma_range[1] > 0 else 0.0
    coma_y = np.random.uniform(*coma_range) if coma_range[1] > 0 else 0.0

    # Random orientation for directional aberrations (astigmatism, coma)
    # Rotate the aberration vector randomly
    angle = np.random.uniform(0, 2 * np.pi)
    astig_mag = np.sqrt(astigmatism_x**2 + astigmatism_y**2)
    coma_mag = np.sqrt(coma_x**2 + coma_y**2)

    astigmatism_x = astig_mag * np.cos(angle)
    astigmatism_y = astig_mag * np.sin(angle)
    coma_x = coma_mag * np.cos(angle)
    coma_y = coma_mag * np.sin(angle)

    # Create aberrated PSF
    return simulation.create_psf_model(
        fwhm=fwhm,
        psf_type='gaussian',
        defocus=defocus,
        astigmatism_x=astigmatism_x,
        astigmatism_y=astigmatism_y,
        coma_x=coma_x,
        coma_y=coma_y,
        wavelength=550e-9,  # V-band
    )


def train_classifier(args):
    """Train a real-bogus classifier on simulated data."""
    print("=" * 80)
    print("Training Real-Bogus Classifier")
    print("=" * 80)

    # Determine real source types
    if args.include_galaxies:
        real_source_types = ['star', 'galaxy']
        print("\nMode: Standard classifier (stars + galaxies as real)")
    else:
        real_source_types = ['star']
        print("\nMode: Stars-only classifier (galaxies as bogus)")

    # Training parameters
    print("\nTraining Configuration:")
    print(f"  Number of simulated images: {args.n_images}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  FWHM range: {args.fwhm_min:.1f} - {args.fwhm_max:.1f} pixels")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Validation split: {args.val_split:.2f}")
    print(f"  Real source types: {real_source_types}")
    print(f"  Asinh softening: {args.asinh_softening_sigma:.2f} sigma")

    # Aberration parameters
    if args.enable_aberrations:
        print(f"\nOptical Aberrations: ENABLED")
        print(f"  Defocus range: {args.defocus_min:.2f} - {args.defocus_max:.2f} waves")
        print(f"  Astigmatism range: {args.astigmatism_min:.2f} - {args.astigmatism_max:.2f} waves")
        print(f"  Coma range: {args.coma_min:.2f} - {args.coma_max:.2f} waves")
        print(f"  Aberration fraction: {args.aberration_fraction:.0%} of images")
    else:
        print(f"\nOptical Aberrations: DISABLED")

    # Generate or load training data
    if args.data is None:
        print(f"\nGenerating training data from {args.n_images} simulated images...")
        print("(This may take several minutes)")

        data = simulation.generate_realbogus_training_data(
            n_images=args.n_images,
            image_size=(args.image_size, args.image_size),
            n_stars_range=(args.n_stars_min, args.n_stars_max),
            n_galaxies_range=(args.n_galaxies_min, args.n_galaxies_max),
            fwhm_range=(args.fwhm_min, args.fwhm_max),
            background_range=(args.bg_min, args.bg_max),
            n_cosmic_rays_range=(args.n_cr_min, args.n_cr_max),
            n_hot_pixels_range=(args.n_hot_min, args.n_hot_max),
            real_source_types=real_source_types,
            augment=not args.no_augment,
            aberration_fraction=args.aberration_fraction if args.enable_aberrations else 0.0,
            defocus_range=(args.defocus_min, args.defocus_max) if args.enable_aberrations else (0.0, 0.0),
            astigmatism_range=(args.astigmatism_min, args.astigmatism_max) if args.enable_aberrations else (0.0, 0.0),
            coma_range=(args.coma_min, args.coma_max) if args.enable_aberrations else (0.0, 0.0),
            asinh_softening=args.asinh_softening_sigma,
            verbose=True
        )

        print(f"\nGenerated {len(data['X'])} training samples")
        print(f"  Real sources: {int(np.sum(data['y']))} ({100*np.mean(data['y']):.1f}%)")
        print(f"  Bogus sources: {int(len(data['y']) - np.sum(data['y']))} ({100*(1-np.mean(data['y'])):.1f}%)")

        # Show detection efficiency summary
        if 'detection_efficiency' in data:
            eff_stats = data['detection_efficiency']
            if 'mean' in eff_stats:
                print(f"\nDetection Efficiency:")
                print(f"  Mean: {eff_stats['mean']:.1%}")
                print(f"  Range: {eff_stats['min']:.1%} - {eff_stats['max']:.1%}")
                if eff_stats['n_flux_adjustments'] > 0:
                    print(f"  Flux adjustments: {eff_stats['n_flux_adjustments']}")
                    print(f"  Final star flux range: {eff_stats['final_star_flux_range']}")
                    print(f"  Final galaxy flux range: {eff_stats['final_galaxy_flux_range']}")

        # Optionally save training data
        if args.save_data:
            data_file = Path(args.output).with_suffix('.npz')
            print(f"\nSaving training data to {data_file}")
            save_payload = {
                'X': data['X'],
                'y': data['y'],
            }
            if 'fwhm' in data:
                save_payload['fwhm'] = data['fwhm']
            np.savez_compressed(data_file, **save_payload)
    else:
        print(f"\nLoading training data from {args.data}")
        loaded = np.load(args.data)
        data = {
            'X': loaded['X'],
            'y': loaded['y'],
        }
        if 'fwhm' in loaded:
            data['fwhm'] = loaded['fwhm']
        print(f"Loaded {len(data['X'])} samples")

    # Train model
    print("\n" + "-" * 80)
    print("Training Model")
    print("-" * 80)

    model, history = realbogus.train_realbogus_classifier(
        training_data=data,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight='balanced' if args.class_weight else None,
        model_file=args.output,
        verbose=True
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nModel saved to: {args.output}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    if 'auc' in history.history:
        print(f"Final training AUC: {history.history['auc'][-1]:.4f}")
        print(f"Final validation AUC: {history.history['val_auc'][-1]:.4f}")

    # Plot training history
    if args.plot:
        plot_training_history(history, args.output)

    return model, history


def plot_training_history(history, model_file):
    """Plot and save training history."""
    output_dir = Path(model_file).parent
    plot_file = output_dir / (Path(model_file).stem + '_training_history.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax = axes[0]
    ax.plot(history.history['accuracy'], label='Training')
    ax.plot(history.history['val_accuracy'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1]
    ax.plot(history.history['loss'], label='Training')
    ax.plot(history.history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Model Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {plot_file}")
    plt.close()


def test_single_image(args):
    """Test classifier on a single image with interactive visualization."""
    print("=" * 80)
    print("Testing on Single Image")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from {args.model}")
    model = realbogus.load_realbogus_model(model_file=args.model)

    # Load or simulate image
    if args.image is not None:
        print(f"\nLoading image from {args.image}")
        from astropy.io import fits
        with fits.open(args.image) as hdul:
            image = hdul[0].data.astype(np.float64)
        truth_catalog = None
    else:
        print("\nSimulating test image...")

        # Create PSF model (with aberrations if enabled)
        if args.enable_aberrations:
            print("  Creating PSF with optical aberrations...")
            aberration_params = {
                'defocus_range': (args.defocus_min, args.defocus_max),
                'astigmatism_range': (args.astigmatism_min, args.astigmatism_max),
                'coma_range': (args.coma_min, args.coma_max),
                'fraction': 1.0  # Always use aberrations for test
            }
            psf_model = create_aberrated_psf_model(
                args.fwhm,
                enable_aberrations=True,
                aberration_params=aberration_params
            )
            if 'aberrations' in psf_model:
                print(f"    Defocus: {psf_model['aberrations']['defocus']:.2f} waves")
                print(f"    Astigmatism: ({psf_model['aberrations']['astigmatism_x']:.2f}, "
                      f"{psf_model['aberrations']['astigmatism_y']:.2f}) waves")
                print(f"    Coma: ({psf_model['aberrations']['coma_x']:.2f}, "
                      f"{psf_model['aberrations']['coma_y']:.2f}) waves")
        else:
            psf_model = 'gaussian'  # Use default

        sim = simulation.simulate_image(
            width=args.image_size,
            height=args.image_size,
            n_stars=50,
            star_fwhm=args.fwhm,
            star_psf=psf_model,
            n_galaxies=15,
            n_cosmic_rays=10,
            n_hot_pixels=20,
            background=1000.0,
            verbose=True
        )
        image = sim['image']
        truth_catalog = sim['catalog']

    # Detect objects
    print("\nDetecting objects...")
    obj_all = photometry.get_objects_sep(
        image,
        thresh=args.threshold,
        aper=2.5 * args.fwhm,
        verbose=True
    )
    print(f"Detected {len(obj_all)} objects")

    # Classify
    print("\nClassifying detections...")
    obj_classified = realbogus.classify_realbogus(
        obj_all,
        image,
        model=model,
        threshold=args.rb_threshold,
        add_score=True,
        flag_bogus=False,  # Keep all for visualization
        asinh_softening=args.asinh_softening_sigma,
        verbose=True
    )

    # Filter for real sources
    obj_real = obj_classified[obj_classified['rb_score'] > args.rb_threshold]

    print(f"\nClassification Results:")
    print(f"  Total detections: {len(obj_all)}")
    print(f"  Classified as real (score > {args.rb_threshold}): {len(obj_real)}")
    print(f"  Classified as bogus: {len(obj_all) - len(obj_real)}")

    # Score statistics
    scores = obj_classified['rb_score']
    print(f"\nReal-Bogus Score Statistics:")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Median: {np.median(scores):.3f}")
    print(f"  Std: {np.std(scores):.3f}")
    print(f"  Min: {np.min(scores):.3f}")
    print(f"  Max: {np.max(scores):.3f}")

    # Visualize
    if args.interactive:
        matplotlib.use('TkAgg')  # Interactive backend

    visualize_classification(
        image,
        obj_all,
        obj_classified,
        obj_real,
        truth_catalog,
        args.rb_threshold,
        args.output
    )

    if args.interactive:
        plt.show()


def visualize_classification(image, obj_all, obj_classified, obj_real, truth_catalog,
                            threshold, output_file):
    """Create comprehensive visualization of classification results."""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. All detections
    ax = fig.add_subplot(gs[0, 0])
    vmin, vmax = np.percentile(image[np.isfinite(image)], [1, 99])
    ax.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax.scatter(obj_all['x'], obj_all['y'], s=80, facecolors='none',
               edgecolors='red', linewidth=1.5, alpha=0.7, label='All')
    ax.set_title(f'All Detections ({len(obj_all)})', fontsize=12, fontweight='bold')
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.legend(loc='upper right')

    # 2. Classified as real
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax.scatter(obj_real['x'], obj_real['y'], s=80, facecolors='none',
               edgecolors='green', linewidth=1.5, alpha=0.8, label='Real')
    ax.set_title(f'Real Sources ({len(obj_real)})', fontsize=12, fontweight='bold')
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.legend(loc='upper right')

    # 3. Classified as bogus
    ax = fig.add_subplot(gs[0, 2])
    obj_bogus = obj_classified[obj_classified['rb_score'] <= threshold]
    ax.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax.scatter(obj_bogus['x'], obj_bogus['y'], s=80, facecolors='none',
               edgecolors='orange', linewidth=1.5, alpha=0.7, label='Bogus')
    ax.set_title(f'Bogus Sources ({len(obj_bogus)})', fontsize=12, fontweight='bold')
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    ax.legend(loc='upper right')

    # 4. Score distribution histogram
    ax = fig.add_subplot(gs[1, 0])
    scores = obj_classified['rb_score']
    ax.hist(scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.2f}')
    ax.set_xlabel('Real-Bogus Score')
    ax.set_ylabel('Number of Detections')
    ax.set_title('Score Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Score vs Flux
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(obj_classified['flux'], obj_classified['rb_score'],
               alpha=0.6, s=30, c=obj_classified['rb_score'],
               cmap='RdYlGn', vmin=0, vmax=1)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Flux [ADU]')
    ax.set_ylabel('Real-Bogus Score')
    ax.set_title('Score vs Flux', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # 6. Score vs FWHM
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(obj_classified['fwhm'], obj_classified['rb_score'],
               alpha=0.6, s=30, c=obj_classified['rb_score'],
               cmap='RdYlGn', vmin=0, vmax=1)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('FWHM [pixels]')
    ax.set_ylabel('Real-Bogus Score')
    ax.set_title('Score vs FWHM', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 7. Truth comparison (if available)
    ax = fig.add_subplot(gs[2, 0])
    if truth_catalog is not None:
        n_stars = len(truth_catalog[truth_catalog['type'] == 'star'])
        n_gal = len(truth_catalog[truth_catalog['type'] == 'galaxy'])
        n_artifacts = len(truth_catalog) - n_stars - n_gal

        categories = ['Stars', 'Galaxies', 'Artifacts', 'Detected\n(Real)', 'Detected\n(Bogus)']
        counts = [n_stars, n_gal, n_artifacts, len(obj_real), len(obj_bogus)]
        colors = ['blue', 'cyan', 'orange', 'green', 'red']

        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('Source Type Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No truth catalog available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, style='italic')
        ax.set_xticks([])
        ax.set_yticks([])

    # 8. Cumulative score distribution
    ax = fig.add_subplot(gs[2, 1])
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cumulative, linewidth=2, color='steelblue')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.2f}')
    ax.set_xlabel('Real-Bogus Score')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. Statistics summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    stats_text = f"""
    Detection Statistics:
    ─────────────────────
    Total detections: {len(obj_all)}
    Real (score > {threshold}): {len(obj_real)}
    Bogus (score ≤ {threshold}): {len(obj_bogus)}

    Score Statistics:
    ─────────────────────
    Mean: {np.mean(scores):.3f}
    Median: {np.median(scores):.3f}
    Std: {np.std(scores):.3f}

    Percentiles:
    ─────────────────────
    10%: {np.percentile(scores, 10):.3f}
    25%: {np.percentile(scores, 25):.3f}
    75%: {np.percentile(scores, 75):.3f}
    90%: {np.percentile(scores, 90):.3f}
    """

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Real-Bogus Classification Results',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save
    if output_file:
        output_path = Path(output_file)
        if output_path.suffix == '':
            output_path = output_path.with_suffix('.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")


def evaluate_classifier(args):
    """Evaluate classifier on multiple images and compute metrics."""
    print("=" * 80)
    print("Evaluating Classifier Performance")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from {args.model}")
    model = realbogus.load_realbogus_model(model_file=args.model)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Evaluation loop
    print(f"\nEvaluating on {args.n_images} simulated images...")

    if args.enable_aberrations:
        print(f"Optical aberrations: ENABLED ({args.aberration_fraction:.0%} of images)")
        print(f"  Defocus: {args.defocus_min:.2f}-{args.defocus_max:.2f} waves")
        print(f"  Astigmatism: {args.astigmatism_min:.2f}-{args.astigmatism_max:.2f} waves")
        print(f"  Coma: {args.coma_min:.2f}-{args.coma_max:.2f} waves")
    else:
        print("Optical aberrations: DISABLED")

    all_metrics = []
    all_scores_real = []
    all_scores_bogus = []
    sample_data = []  # Store sample images for interactive display

    # Detection efficiency tracking
    detection_efficiencies = []
    n_flux_adjustments = 0
    low_efficiency_count = 0

    # Dynamic flux ranges (will be adjusted based on detection efficiency)
    star_flux_range = (100, 10000)
    galaxy_flux_range = (500, 5000)

    for i in range(args.n_images):
        # Random FWHM for this image
        fwhm = np.random.uniform(args.fwhm_min, args.fwhm_max)

        # Create PSF model (with aberrations if enabled)
        if args.enable_aberrations:
            aberration_params = {
                'defocus_range': (args.defocus_min, args.defocus_max),
                'astigmatism_range': (args.astigmatism_min, args.astigmatism_max),
                'coma_range': (args.coma_min, args.coma_max),
                'fraction': args.aberration_fraction
            }
            psf_model = create_aberrated_psf_model(
                fwhm,
                enable_aberrations=True,
                aberration_params=aberration_params
            )
        else:
            psf_model = 'gaussian'  # Use default

        # Simulate image (with dynamic flux ranges)
        sim = simulation.simulate_image(
            width=args.image_size,
            height=args.image_size,
            n_stars=np.random.randint(args.n_stars_min, args.n_stars_max + 1),
            star_flux_range=star_flux_range,
            star_fwhm=fwhm,
            star_psf=psf_model,
            n_galaxies=np.random.randint(args.n_galaxies_min, args.n_galaxies_max + 1),
            galaxy_flux_range=galaxy_flux_range,
            n_cosmic_rays=np.random.randint(args.n_cr_min, args.n_cr_max + 1),
            n_hot_pixels=np.random.randint(args.n_hot_min, args.n_hot_max + 1),
            background=10**np.random.uniform(np.log10(args.bg_min), np.log10(args.bg_max)),
            verbose=False
        )

        truth = sim['catalog']

        # Detect objects
        try:
            obj = photometry.get_objects_sep(
                sim['image'],
                thresh=args.threshold,
                verbose=False
            )
        except Exception as e:
            print(f"  Warning: Detection failed for image {i+1}: {e}")
            continue

        if len(obj) == 0:
            continue

        # Classify
        obj_scored = realbogus.classify_realbogus(
            obj,
            sim['image'],
            model=model,
            add_score=True,
            flag_bogus=False,
            asinh_softening=args.asinh_softening_sigma,
            verbose=False
        )

        # Calculate detection efficiency (before computing metrics)
        # Count real sources in truth catalog (stars and galaxies, not artifacts)
        truth_real = truth[(truth['type'] == 'star') | (truth['type'] == 'galaxy')]
        n_simulated_real = len(truth_real)

        # Match detections to real sources
        n_detected_real = 0
        for obj in obj_scored:
            dx = truth_real['x'] - obj['x']
            dy = truth_real['y'] - obj['y']
            dist = np.sqrt(dx**2 + dy**2)
            if len(dist) > 0 and np.min(dist) < args.match_radius:
                n_detected_real += 1

        if n_simulated_real > 0:
            detection_efficiency = n_detected_real / n_simulated_real
            detection_efficiencies.append(detection_efficiency)

            # Check if efficiency is too low
            if detection_efficiency < args.min_detection_efficiency:
                low_efficiency_count += 1

                # If efficiency has been low for 3 consecutive images, adjust flux
                if (low_efficiency_count >= 3 and
                    n_flux_adjustments < args.max_flux_adjustments):

                    old_star_min = star_flux_range[0]
                    old_galaxy_min = galaxy_flux_range[0]

                    # Increase minimum flux to improve detection rate
                    star_flux_range = (
                        star_flux_range[0] * args.flux_adjustment_factor,
                        star_flux_range[1]
                    )
                    galaxy_flux_range = (
                        galaxy_flux_range[0] * args.flux_adjustment_factor,
                        galaxy_flux_range[1]
                    )

                    n_flux_adjustments += 1
                    low_efficiency_count = 0  # Reset counter

                    print(f"\n  ⚠️  Detection efficiency low for 3 images - adjusting flux ranges:")
                    print(f"      Stars: {old_star_min:.0f}-{star_flux_range[1]:.0f} → "
                          f"{star_flux_range[0]:.0f}-{star_flux_range[1]:.0f}")
                    print(f"      Galaxies: {old_galaxy_min:.0f}-{galaxy_flux_range[1]:.0f} → "
                          f"{galaxy_flux_range[0]:.0f}-{galaxy_flux_range[1]:.0f}\n")
            else:
                low_efficiency_count = 0  # Reset if efficiency is good

        # Match to truth for metrics
        metrics = compute_metrics(obj_scored, truth, args.rb_threshold, args.match_radius)

        # Add detection efficiency to metrics
        if n_simulated_real > 0:
            metrics['detection_efficiency'] = detection_efficiency
            metrics['n_simulated_real'] = n_simulated_real
            metrics['n_detected_real'] = n_detected_real

        all_metrics.append(metrics)

        # Collect scores by true class
        all_scores_real.extend(metrics['scores_real'])
        all_scores_bogus.extend(metrics['scores_bogus'])

        # Store sample for interactive visualization
        if args.interactive and len(sample_data) < args.n_interactive:
            sample_data.append({
                'image': sim['image'],
                'objects': obj_scored,
                'truth': truth
            })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{args.n_images} images...")

    # Print detection efficiency summary
    if detection_efficiencies:
        mean_efficiency = np.mean(detection_efficiencies)
        median_efficiency = np.median(detection_efficiencies)
        min_efficiency = np.min(detection_efficiencies)
        max_efficiency = np.max(detection_efficiencies)

        print("\n" + "=" * 80)
        print("Detection Efficiency Summary")
        print("=" * 80)
        print(f"  Mean:   {mean_efficiency:.1%}")
        print(f"  Median: {median_efficiency:.1%}")
        print(f"  Range:  {min_efficiency:.1%} - {max_efficiency:.1%}")
        if n_flux_adjustments > 0:
            print(f"  Flux adjustments made: {n_flux_adjustments}")
            print(f"  Final star flux range: {star_flux_range[0]:.0f} - {star_flux_range[1]:.0f}")
            print(f"  Final galaxy flux range: {galaxy_flux_range[0]:.0f} - {galaxy_flux_range[1]:.0f}")

    # Aggregate metrics
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    aggregate_metrics = aggregate_results(all_metrics)

    # Add detection efficiency to aggregate metrics
    if detection_efficiencies:
        aggregate_metrics['detection_efficiency_mean'] = np.mean(detection_efficiencies)
        aggregate_metrics['detection_efficiency_std'] = np.std(detection_efficiencies)
        aggregate_metrics['detection_efficiency_median'] = np.median(detection_efficiencies)
        aggregate_metrics['n_flux_adjustments'] = n_flux_adjustments
        aggregate_metrics['final_star_flux_range'] = star_flux_range
        aggregate_metrics['final_galaxy_flux_range'] = galaxy_flux_range

    print(f"\nPerformance Metrics (threshold = {args.rb_threshold}):")
    print(f"  Precision: {aggregate_metrics['precision']:.3f} ± {aggregate_metrics['precision_std']:.3f}")
    print(f"  Recall: {aggregate_metrics['recall']:.3f} ± {aggregate_metrics['recall_std']:.3f}")
    print(f"  F1-Score: {aggregate_metrics['f1']:.3f} ± {aggregate_metrics['f1_std']:.3f}")
    print(f"  Accuracy: {aggregate_metrics['accuracy']:.3f} ± {aggregate_metrics['accuracy_std']:.3f}")

    print(f"\nConfusion Matrix (mean):")
    print(f"                Predicted")
    print(f"              Real    Bogus")
    print(f"  Actual Real  {aggregate_metrics['tp']:.1f}    {aggregate_metrics['fn']:.1f}")
    print(f"        Bogus  {aggregate_metrics['fp']:.1f}    {aggregate_metrics['tn']:.1f}")

    # Save metrics (convert numpy types to native Python types for JSON serialization)
    metrics_file = output_dir / 'metrics.json'
    metrics_to_save = {}
    for key, value in aggregate_metrics.items():
        # Convert numpy types to native Python types
        if isinstance(value, (np.integer, np.floating)):
            metrics_to_save[key] = value.item()
        elif isinstance(value, np.ndarray):
            metrics_to_save[key] = value.tolist()
        else:
            metrics_to_save[key] = value

    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    # Plot results
    plot_evaluation_results(
        all_metrics,
        all_scores_real,
        all_scores_bogus,
        args.rb_threshold,
        output_dir
    )

    # Interactive visualization if requested
    if args.interactive and len(sample_data) > 0:
        print(f"\n" + "=" * 80)
        print(f"Interactive Visualization ({len(sample_data)} samples)")
        print("=" * 80)
        print("\nDisplaying sample images with real/bogus markings...")
        print("Close each window to proceed to the next image.\n")
        visualize_interactive_samples(sample_data, args.rb_threshold, output_dir)

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


def compute_metrics(obj_scored, truth_catalog, threshold, match_radius=3.0):
    """Compute classification metrics for one image."""

    # Determine true labels by matching to truth
    true_labels = []
    scores = []

    # Match each detection to truth
    for obj in obj_scored:
        dx = truth_catalog['x'] - obj['x']
        dy = truth_catalog['y'] - obj['y']
        dist = np.sqrt(dx**2 + dy**2)

        if len(dist) > 0 and np.min(dist) < match_radius:
            # Matched - check if it's a real source
            idx = np.argmin(dist)
            true_type = truth_catalog['type'][idx]
            is_real = (true_type == 'star') or (true_type == 'galaxy')
            true_labels.append(1 if is_real else 0)
        else:
            # Unmatched - spurious detection
            true_labels.append(0)

        scores.append(obj['rb_score'])

    true_labels = np.array(true_labels)
    scores = np.array(scores)
    pred_labels = (scores > threshold).astype(int)

    # Compute confusion matrix
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    tn = np.sum((pred_labels == 0) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0.0

    # Separate scores by true class
    scores_real = scores[true_labels == 1].tolist() if np.any(true_labels == 1) else []
    scores_bogus = scores[true_labels == 0].tolist() if np.any(true_labels == 0) else []

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'scores_real': scores_real,
        'scores_bogus': scores_bogus,
        'n_detections': len(obj_scored),
        'n_real_truth': np.sum(true_labels == 1),
        'n_bogus_truth': np.sum(true_labels == 0)
    }


def aggregate_results(all_metrics):
    """Aggregate metrics across multiple images."""

    if len(all_metrics) == 0:
        return {}

    # Extract arrays
    precisions = [m['precision'] for m in all_metrics]
    recalls = [m['recall'] for m in all_metrics]
    f1s = [m['f1'] for m in all_metrics]
    accuracies = [m['accuracy'] for m in all_metrics]

    # Confusion matrix sums
    tp = sum(m['tp'] for m in all_metrics)
    fp = sum(m['fp'] for m in all_metrics)
    tn = sum(m['tn'] for m in all_metrics)
    fn = sum(m['fn'] for m in all_metrics)

    return {
        'precision': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall': np.mean(recalls),
        'recall_std': np.std(recalls),
        'f1': np.mean(f1s),
        'f1_std': np.std(f1s),
        'accuracy': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'n_images': len(all_metrics)
    }


def plot_evaluation_results(all_metrics, scores_real, scores_bogus, threshold, output_dir):
    """Create comprehensive evaluation plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Precision, Recall, F1 across images
    ax = axes[0, 0]
    precisions = [m['precision'] for m in all_metrics]
    recalls = [m['recall'] for m in all_metrics]
    f1s = [m['f1'] for m in all_metrics]

    x = np.arange(len(all_metrics))
    ax.plot(x, precisions, 'o-', label='Precision', alpha=0.7)
    ax.plot(x, recalls, 's-', label='Recall', alpha=0.7)
    ax.plot(x, f1s, '^-', label='F1-Score', alpha=0.7)
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Score')
    ax.set_title('Metrics Across Images')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 2. Score distributions
    ax = axes[0, 1]
    if len(scores_real) > 0:
        ax.hist(scores_real, bins=30, alpha=0.6, label='True Real',
                color='green', edgecolor='black')
    if len(scores_bogus) > 0:
        ax.hist(scores_bogus, bins=30, alpha=0.6, label='True Bogus',
                color='red', edgecolor='black')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.2f}')
    ax.set_xlabel('Real-Bogus Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distribution by True Class')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. ROC-like curve (score threshold vs metrics)
    ax = axes[0, 2]
    thresholds = np.linspace(0, 1, 50)
    precisions_curve = []
    recalls_curve = []

    for thresh in thresholds:
        # Recompute metrics at this threshold
        temp_metrics = []
        for m in all_metrics:
            # Would need to recompute from scores, simplify for now
            pass
        # Simplified: just show the evaluated threshold

    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Current = {threshold:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Metrics vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # 4. Confusion matrix heatmap
    ax = axes[1, 0]
    tp = sum(m['tp'] for m in all_metrics)
    fp = sum(m['fp'] for m in all_metrics)
    tn = sum(m['tn'] for m in all_metrics)
    fn = sum(m['fn'] for m in all_metrics)

    conf_matrix = np.array([[tp, fn], [fp, tn]])
    im = ax.imshow(conf_matrix, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Real', 'Bogus'])
    ax.set_yticklabels(['Real', 'Bogus'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Total)')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{int(conf_matrix[i, j])}',
                          ha="center", va="center", color="black" if conf_matrix[i, j] < conf_matrix.max()/2 else "white",
                          fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax)

    # 5. Metric distributions (box plots)
    ax = axes[1, 1]
    data_to_plot = [precisions, recalls, f1s, [m['accuracy'] for m in all_metrics]]
    bp = ax.boxplot(data_to_plot, labels=['Precision', 'Recall', 'F1', 'Accuracy'],
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Score')
    ax.set_title('Metric Distributions')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    # 6. Cumulative score distributions
    ax = axes[1, 2]
    if len(scores_real) > 0:
        sorted_real = np.sort(scores_real)
        cumulative_real = np.arange(1, len(sorted_real) + 1) / len(sorted_real)
        ax.plot(sorted_real, cumulative_real, linewidth=2,
                label='True Real', color='green')
    if len(scores_bogus) > 0:
        sorted_bogus = np.sort(scores_bogus)
        cumulative_bogus = np.arange(1, len(sorted_bogus) + 1) / len(sorted_bogus)
        ax.plot(sorted_bogus, cumulative_bogus, linewidth=2,
                label='True Bogus', color='red')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.2f}')
    ax.set_xlabel('Real-Bogus Score')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    plot_file = output_dir / 'evaluation_results.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Evaluation plots saved to: {plot_file}")
    plt.close()


def visualize_interactive_samples(sample_data, threshold, output_dir):
    """
    Create interactive visualizations of sample images with real/bogus markings.

    Parameters
    ----------
    sample_data : list of dict
        List of sample dictionaries with 'image', 'objects', 'truth' keys
    threshold : float
        Real-bogus classification threshold
    output_dir : Path
        Directory to save plots
    """
    import matplotlib
    matplotlib.use('TkAgg')  # Interactive backend
    import matplotlib.pyplot as plt

    for idx, sample in enumerate(sample_data):
        image = sample['image']
        obj = sample['objects']
        truth = sample['truth']

        # Separate real and bogus based on scores
        real_mask = obj['rb_score'] > threshold
        bogus_mask = ~real_mask

        obj_real = obj[real_mask]
        obj_bogus = obj[bogus_mask]

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Sample Image {idx+1}', fontsize=14, fontweight='bold')

        # Compute image display range (robust scaling)
        vmin, vmax = np.percentile(image, [1, 99])

        # 1. All detections
        ax = axes[0]
        ax.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        ax.scatter(obj['x'], obj['y'], s=80, facecolors='none',
                   edgecolors='yellow', linewidth=1.5, alpha=0.8, label='All')
        # Mark truth positions
        if truth is not None and len(truth) > 0:
            truth_stars = truth[truth['type'] == 'star']
            if len(truth_stars) > 0:
                ax.scatter(truth_stars['x'], truth_stars['y'], s=40,
                           marker='x', c='cyan', linewidth=1, alpha=0.5, label='True Stars')
        ax.set_title(f'All Detections ({len(obj)})')
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.legend(loc='upper right', fontsize=8)

        # 2. Real vs Bogus
        ax = axes[1]
        ax.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        if len(obj_real) > 0:
            ax.scatter(obj_real['x'], obj_real['y'], s=80, facecolors='none',
                       edgecolors='green', linewidth=1.5, alpha=0.8, label='Real')
        if len(obj_bogus) > 0:
            ax.scatter(obj_bogus['x'], obj_bogus['y'], s=80, facecolors='none',
                       edgecolors='red', linewidth=1.5, alpha=0.8, label='Bogus')
        ax.set_title(f'Classified: {len(obj_real)} Real, {len(obj_bogus)} Bogus')
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.legend(loc='upper right', fontsize=8)

        # 3. Score distribution
        ax = axes[2]
        scores = obj['rb_score'][~np.isnan(obj['rb_score'])]
        if len(scores) > 0:
            ax.hist(scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold = {threshold:.2f}')
            ax.set_xlabel('Real-Bogus Score')
            ax.set_ylabel('Number of Detections')
            ax.set_title('Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No valid scores', ha='center', va='center',
                    transform=ax.transAxes)

        plt.tight_layout()

        # Save
        plot_file = output_dir / f'interactive_sample_{idx+1}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"  Sample {idx+1} saved to: {plot_file}")

        # Show interactively
        plt.show()

    # Switch back to non-interactive backend
    matplotlib.use('Agg')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Real-Bogus Classifier Training and Evaluation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new classifier (stars-only by default)
  %(prog)s train --n-images 500 --epochs 30 --output model.h5

  # Train classifier that includes galaxies as real sources
  %(prog)s train --include-galaxies --n-images 500 --output full_model.h5

  # Test with optical aberrations (interactive)
  %(prog)s test --model model.h5 --enable-aberrations --interactive

  # Evaluate with aberrated PSFs (30%% of images have aberrations)
  %(prog)s evaluate --model model.h5 --n-images 50 --enable-aberrations \\
    --aberration-fraction 0.3 --output results/

  # Evaluate robustness to strong aberrations
  %(prog)s evaluate --model model.h5 --enable-aberrations \\
    --defocus-max 3.0 --astigmatism-max 2.0 --coma-max 1.5 --output results_aberrated/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True

    # ========================================================================
    # Train command
    # ========================================================================
    train_parser = subparsers.add_parser('train', help='Train a new classifier')

    # Training data
    train_parser.add_argument('--n-images', type=int, default=500,
                            help='Number of simulated images (default: 500)')
    train_parser.add_argument('--image-size', type=int, default=1024,
                            help='Simulated image size (default: 1024)')
    train_parser.add_argument('--data', type=str,
                            help='Load training data from .npz file instead of generating')
    train_parser.add_argument('--save-data', action='store_true',
                            help='Save generated training data to .npz file')

    # Source parameters
    train_parser.add_argument('--include-galaxies', action='store_true',
                            help='Include galaxies as real sources (default: stars-only)')
    train_parser.add_argument('--n-stars-min', type=int, default=30,
                            help='Minimum stars per image (default: 30)')
    train_parser.add_argument('--n-stars-max', type=int, default=80,
                            help='Maximum stars per image (default: 80)')
    train_parser.add_argument('--n-galaxies-min', type=int, default=10,
                            help='Minimum galaxies per image (default: 10)')
    train_parser.add_argument('--n-galaxies-max', type=int, default=30,
                            help='Maximum galaxies per image (default: 30)')
    train_parser.add_argument('--fwhm-min', type=float, default=2.0,
                            help='Minimum FWHM in pixels (default: 2.0)')
    train_parser.add_argument('--fwhm-max', type=float, default=6.0,
                            help='Maximum FWHM in pixels (default: 6.0)')
    train_parser.add_argument('--asinh-softening-sigma', type=float,
                            default=realbogus.DEFAULT_ASINH_SOFTENING_SIGMA,
                            help='Asinh softening in units of background sigma')

    # Artifact parameters
    train_parser.add_argument('--n-cr-min', type=int, default=5,
                            help='Minimum cosmic rays per image (default: 5)')
    train_parser.add_argument('--n-cr-max', type=int, default=20,
                            help='Maximum cosmic rays per image (default: 20)')
    train_parser.add_argument('--n-hot-min', type=int, default=10,
                            help='Minimum hot pixels per image (default: 10)')
    train_parser.add_argument('--n-hot-max', type=int, default=30,
                            help='Maximum hot pixels per image (default: 30)')
    train_parser.add_argument('--bg-min', type=float, default=100,
                            help='Minimum background level (default: 100)')
    train_parser.add_argument('--bg-max', type=float, default=10000,
                            help='Maximum background level (default: 10000)')

    # Training parameters
    train_parser.add_argument('--epochs', type=int, default=30,
                            help='Training epochs (default: 30)')
    train_parser.add_argument('--batch-size', type=int, default=64,
                            help='Batch size (default: 64)')
    train_parser.add_argument('--val-split', type=float, default=0.15,
                            help='Validation split fraction (default: 0.15)')
    train_parser.add_argument('--no-class-weight', dest='class_weight',
                            action='store_false',
                            help='Disable balanced class weighting')
    train_parser.add_argument('--no-augment', action='store_true',
                            help='Disable data augmentation')

    # Optical aberrations
    train_parser.add_argument('--enable-aberrations', action='store_true',
                            help='Enable optical aberrations in simulated PSFs')
    train_parser.add_argument('--defocus-min', type=float, default=0.0,
                            help='Minimum defocus aberration in waves (default: 0.0)')
    train_parser.add_argument('--defocus-max', type=float, default=2.0,
                            help='Maximum defocus aberration in waves (default: 2.0)')
    train_parser.add_argument('--astigmatism-min', type=float, default=0.0,
                            help='Minimum astigmatism aberration in waves (default: 0.0)')
    train_parser.add_argument('--astigmatism-max', type=float, default=1.5,
                            help='Maximum astigmatism aberration in waves (default: 1.5)')
    train_parser.add_argument('--coma-min', type=float, default=0.0,
                            help='Minimum coma aberration in waves (default: 0.0)')
    train_parser.add_argument('--coma-max', type=float, default=1.0,
                            help='Maximum coma aberration in waves (default: 1.0)')
    train_parser.add_argument('--aberration-fraction', type=float, default=0.3,
                            help='Fraction of images with aberrations (default: 0.3)')

    # Output
    train_parser.add_argument('--output', type=str, default='realbogus_model.h5',
                            help='Output model file (default: realbogus_model.h5)')
    train_parser.add_argument('--plot', action='store_true',
                            help='Plot training history')

    # ========================================================================
    # Test command
    # ========================================================================
    test_parser = subparsers.add_parser('test', help='Test on single image')

    test_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model file (.h5)')
    test_parser.add_argument('--image', type=str,
                           help='FITS image to test on (if not provided, simulates one)')
    test_parser.add_argument('--image-size', type=int, default=1024,
                           help='Simulated image size if no image provided (default: 1024)')
    test_parser.add_argument('--fwhm', type=float, default=3.5,
                           help='FWHM for detection/simulation (default: 3.5)')
    test_parser.add_argument('--asinh-softening-sigma', type=float,
                           default=realbogus.DEFAULT_ASINH_SOFTENING_SIGMA,
                           help='Asinh softening in units of background sigma')
    test_parser.add_argument('--threshold', type=float, default=3.0,
                           help='Detection threshold in sigma (default: 3.0)')
    test_parser.add_argument('--rb-threshold', type=float, default=0.5,
                           help='Real-bogus classification threshold (default: 0.5)')

    # Optical aberrations
    test_parser.add_argument('--enable-aberrations', action='store_true',
                           help='Enable optical aberrations in simulated PSF')
    test_parser.add_argument('--defocus-min', type=float, default=0.0,
                           help='Minimum defocus aberration in waves (default: 0.0)')
    test_parser.add_argument('--defocus-max', type=float, default=2.0,
                           help='Maximum defocus aberration in waves (default: 2.0)')
    test_parser.add_argument('--astigmatism-min', type=float, default=0.0,
                           help='Minimum astigmatism aberration in waves (default: 0.0)')
    test_parser.add_argument('--astigmatism-max', type=float, default=1.5,
                           help='Maximum astigmatism aberration in waves (default: 1.5)')
    test_parser.add_argument('--coma-min', type=float, default=0.0,
                           help='Minimum coma aberration in waves (default: 0.0)')
    test_parser.add_argument('--coma-max', type=float, default=1.0,
                           help='Maximum coma aberration in waves (default: 1.0)')

    # Output
    test_parser.add_argument('--interactive', action='store_true',
                           help='Show interactive plot window')
    test_parser.add_argument('--output', type=str, default='test_result.png',
                           help='Output plot file (default: test_result.png)')

    # ========================================================================
    # Evaluate command
    # ========================================================================
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate on multiple images')

    eval_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model file (.h5)')
    eval_parser.add_argument('--n-images', type=int, default=50,
                           help='Number of test images (default: 50)')
    eval_parser.add_argument('--image-size', type=int, default=1024,
                           help='Test image size (default: 1024)')
    eval_parser.add_argument('--fwhm-min', type=float, default=2.0,
                           help='Minimum FWHM (default: 2.0)')
    eval_parser.add_argument('--fwhm-max', type=float, default=6.0,
                           help='Maximum FWHM (default: 6.0)')
    eval_parser.add_argument('--asinh-softening-sigma', type=float,
                           default=realbogus.DEFAULT_ASINH_SOFTENING_SIGMA,
                           help='Asinh softening in units of background sigma')
    eval_parser.add_argument('--threshold', type=float, default=3.0,
                           help='Detection threshold in sigma (default: 3.0)')
    eval_parser.add_argument('--rb-threshold', type=float, default=0.5,
                           help='Real-bogus classification threshold (default: 0.5)')
    eval_parser.add_argument('--match-radius', type=float, default=3.0,
                           help='Truth matching radius in pixels (default: 3.0)')
    eval_parser.add_argument('--output', type=str, default='evaluation/',
                           help='Output directory (default: evaluation/)')

    # Simulation parameters
    eval_parser.add_argument('--n-stars-min', type=int, default=30,
                           help='Minimum stars per image (default: 30)')
    eval_parser.add_argument('--n-stars-max', type=int, default=80,
                           help='Maximum stars per image (default: 80)')
    eval_parser.add_argument('--n-galaxies-min', type=int, default=10,
                           help='Minimum galaxies per image (default: 10)')
    eval_parser.add_argument('--n-galaxies-max', type=int, default=30,
                           help='Maximum galaxies per image (default: 30)')
    eval_parser.add_argument('--n-cr-min', type=int, default=5,
                           help='Minimum cosmic rays per image (default: 5)')
    eval_parser.add_argument('--n-cr-max', type=int, default=20,
                           help='Maximum cosmic rays per image (default: 20)')
    eval_parser.add_argument('--n-hot-min', type=int, default=10,
                           help='Minimum hot pixels per image (default: 10)')
    eval_parser.add_argument('--n-hot-max', type=int, default=30,
                           help='Maximum hot pixels per image (default: 30)')
    eval_parser.add_argument('--bg-min', type=float, default=100,
                           help='Minimum background level (default: 100)')
    eval_parser.add_argument('--bg-max', type=float, default=10000,
                           help='Maximum background level (default: 10000)')

    # Optical aberrations
    eval_parser.add_argument('--enable-aberrations', action='store_true',
                           help='Enable optical aberrations in simulated PSFs')
    eval_parser.add_argument('--defocus-min', type=float, default=0.0,
                           help='Minimum defocus aberration in waves (default: 0.0)')
    eval_parser.add_argument('--defocus-max', type=float, default=2.0,
                           help='Maximum defocus aberration in waves (default: 2.0)')
    eval_parser.add_argument('--astigmatism-min', type=float, default=0.0,
                           help='Minimum astigmatism aberration in waves (default: 0.0)')
    eval_parser.add_argument('--astigmatism-max', type=float, default=1.5,
                           help='Maximum astigmatism aberration in waves (default: 1.5)')
    eval_parser.add_argument('--coma-min', type=float, default=0.0,
                           help='Minimum coma aberration in waves (default: 0.0)')
    eval_parser.add_argument('--coma-max', type=float, default=1.0,
                           help='Maximum coma aberration in waves (default: 1.0)')
    eval_parser.add_argument('--aberration-fraction', type=float, default=0.3,
                           help='Fraction of images with aberrations (default: 0.3)')

    # Detection efficiency diagnostics
    eval_parser.add_argument('--min-detection-efficiency', type=float, default=0.3,
                           help='Minimum detection efficiency (0-1) before flux adjustment (default: 0.3)')
    eval_parser.add_argument('--flux-adjustment-factor', type=float, default=1.5,
                           help='Factor to multiply min flux by when adjusting (default: 1.5)')
    eval_parser.add_argument('--max-flux-adjustments', type=int, default=3,
                           help='Maximum number of flux adjustments (default: 3)')

    # Interactive visualization
    eval_parser.add_argument('--interactive', action='store_true',
                           help='Show interactive visualization of sample images')
    eval_parser.add_argument('--n-interactive', type=int, default=3,
                           help='Number of images to show interactively (default: 3)')

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == 'train':
        train_classifier(args)
    elif args.command == 'test':
        test_single_image(args)
    elif args.command == 'evaluate':
        evaluate_classifier(args)


if __name__ == '__main__':
    main()
