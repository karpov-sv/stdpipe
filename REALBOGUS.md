# Real-Bogus Classification in STDPipe

Comprehensive guide to CNN-based real-bogus classification for astronomical object detection.

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Default Behavior: Stars-Only Classification](#default-behavior-stars-only-classification)
4. [Implementation](#implementation)
5. [Command-Line Tool](#command-line-tool)
6. [Python API](#python-api)
7. [Training](#training)
8. [Testing](#testing)
9. [Evaluation](#evaluation)
10. [Use Cases](#use-cases)
11. [Tips and Best Practices](#tips-and-best-practices)
12. [Advanced Features](#advanced-features)
13. [Implementation History](#implementation-history)

---

## Overview

Real-bogus classification is a machine learning technique to distinguish genuine astronomical sources from artifacts (cosmic rays, hot pixels, satellite trails, bad columns, etc.) in automated detection pipelines. STDPipe includes a CNN-based classifier that can be trained on simulated images and applied to real observations.

### What is Real-Bogus Classification?

After detecting objects in astronomical images, many detections are not real astronomical sources but rather instrumental artifacts or noise fluctuations. A real-bogus classifier assigns each detection a score (0-1) indicating the probability that it's a real astronomical source.

### Why Real-Bogus Classification?

- **Reduces false positives** in automated pipelines
- **Saves manual inspection time** by pre-filtering obvious artifacts
- **Improves completeness** compared to simple thresholding
- **Adaptable** to different instruments and observing conditions
- **Trainable** on simulated data without labeled real observations

---

## Key Concepts

### Classification Modes

STDPipe supports two classification modes:

| Mode | Real Sources | Bogus Sources | Use Case |
|------|--------------|---------------|----------|
| **Stars-Only (Default)** | Stars only | Galaxies + Artifacts | Transients, variables, astrometry |
| **Full (Opt-in)** | Stars + Galaxies | Artifacts only | Galaxy surveys, general purpose |

**Default behavior** (as of current version): Only stars are treated as real sources. Galaxies are considered bogus along with artifacts.

### Real-Bogus Score

Each detection receives a score between 0 and 1:
- **Score > threshold** → classified as real
- **Score ≤ threshold** → classified as bogus
- **Default threshold**: 0.5
- **Adjustable** based on your science case (higher = more pure, lower = more complete)

### Training Data

The classifier is trained on **simulated images** containing:
- **Real sources**: Stars (point sources) and optionally galaxies (extended sources)
- **Artifacts**: Cosmic rays, hot pixels, satellite trails, bad columns
- **Backgrounds**: Variable sky background levels
- **PSF variation**: Realistic FWHM ranges

---

## Default Behavior: Stars-Only Classification

### Why Stars-Only is the Default

Starting from the current version, **only stars are treated as real sources by default**. This design choice reflects the most common use cases:

1. **Transient Detection**: Reject stationary galaxies, keep moving/variable stars
2. **Variable Star Surveys**: Focus on stellar variability
3. **Astrometry**: Reject extended sources that complicate centroiding
4. **Crowded Field Photometry**: Prioritize stellar photometry

### What Gets Classified

**Real (label = 1):**
- Stars (point sources)

**Bogus (label = 0):**
- Galaxies (extended sources)
- Cosmic rays
- Hot pixels
- Satellite trails
- Bad columns/rows
- Any other artifacts

### Opting In to Include Galaxies

To train a classifier that treats both stars and galaxies as real sources:

**Python API:**
```python
from stdpipe import realbogus

# Include galaxies as real sources
model, history = realbogus.train_realbogus_classifier(
    n_simulated=500,
    real_source_types=['star', 'galaxy'],  # Explicit opt-in
    epochs=30,
    model_file='full_model.h5'
)
```

**Command-Line:**
```bash
# Include galaxies as real sources
python train_realbogus.py train \
    --include-galaxies \
    --n-images 500 \
    --output full_model.h5
```

---

## Implementation

### Architecture

The real-bogus classifier uses a convolutional neural network (CNN) with:

- **Input**: 31×31 pixel cutouts around each detection
- **3 channels**: Original image, background-subtracted, edge-enhanced
- **FWHM feature**: Scalar input encoding the PSF width
- **Architecture**: Multiple conv layers with batch normalization and dropout
- **Output**: Single sigmoid output (probability of being real)

### Model Creation

```python
from stdpipe import realbogus

# Create a new model
model = realbogus.create_realbogus_model(
    input_shape=(31, 31, 3),
    use_fwhm=True,
    n_filters_base=32
)
```

### Cutout Preprocessing

Each detection is converted to a 31×31 pixel cutout with:
1. **Robust normalization**: Clipping and scaling based on percentiles
2. **3-channel representation**:
   - Channel 1: Original image
   - Channel 2: Background-subtracted
   - Channel 3: Edge-enhanced (Sobel filter)
3. **Optional downscaling**: For large cutouts
4. **Padding**: To ensure consistent size

---

## Command-Line Tool

STDPipe includes `train_realbogus.py`, a comprehensive CLI tool for training, testing, and evaluating classifiers.

### Installation

The tool is located in `examples/train_realbogus.py` and can be run directly:

```bash
cd examples
python train_realbogus.py --help
```

### Quick Start

```bash
# Train a classifier (stars-only by default)
python train_realbogus.py train --n-images 500 --epochs 30 --output model.h5

# Test on single image
python train_realbogus.py test --model model.h5 --image test.fits --interactive

# Evaluate performance
python train_realbogus.py evaluate --model model.h5 --n-images 100 --output results/
```

### Commands

The tool provides three main commands:

1. **`train`** - Train a new classifier on simulated data
2. **`test`** - Test on a single image with visualization
3. **`evaluate`** - Evaluate on multiple images with metrics

---

## Python API

### High-Level API

#### Training

```python
from stdpipe import realbogus

# Train on simulated data (stars-only by default)
model, history = realbogus.train_realbogus_classifier(
    n_simulated=500,
    epochs=30,
    model_file='model.h5',
    verbose=True
)
```

#### Classification

```python
from stdpipe import realbogus, photometry
from astropy.io import fits

# Load model
model = realbogus.load_realbogus_model('model.h5')

# Load image and detect objects
image = fits.getdata('observation.fits')
objects = photometry.get_objects_sep(image, thresh=3.0)

# Classify (removes bogus detections)
real_objects = realbogus.classify_realbogus(
    objects,
    image,
    model=model,
    threshold=0.5,
    verbose=True
)

print(f"Found {len(real_objects)} real sources")
```

#### Scoring Only

```python
# Add scores without filtering
objects_scored = realbogus.classify_realbogus(
    objects,
    image,
    model=model,
    add_score=True,
    flag_bogus=False
)

# Scores are in 'rb_score' column
print(objects_scored['rb_score'])
```

### Low-Level API

#### Custom Training Data

```python
from stdpipe import simulation, realbogus

# Generate training data
data = simulation.generate_realbogus_training_data(
    n_images=100,
    image_size=(1024, 1024),
    n_stars_range=(30, 80),
    n_galaxies_range=(10, 30),
    fwhm_range=(2.0, 6.0),
    # real_source_types=['star'] is the default
    augment=True,
    verbose=True
)

print(f"Real sources: {data['y'].sum()}")
print(f"Bogus sources: {len(data['y']) - data['y'].sum()}")

# Train model
model, history = realbogus.train_realbogus_classifier(
    training_data=data,
    epochs=30,
    model_file='custom_model.h5'
)
```

---

## Training

### Basic Training

**Command-Line:**
```bash
# Train stars-only classifier (default)
python train_realbogus.py train \
    --n-images 500 \
    --epochs 30 \
    --output stars_model.h5

# Train with galaxies as real sources
python train_realbogus.py train \
    --include-galaxies \
    --n-images 500 \
    --epochs 30 \
    --output full_model.h5
```

**Python:**
```python
# Stars-only (default)
model, history = realbogus.train_realbogus_classifier(
    n_simulated=500,
    epochs=30,
    model_file='stars_model.h5'
)

# Include galaxies
model, history = realbogus.train_realbogus_classifier(
    n_simulated=500,
    real_source_types=['star', 'galaxy'],
    epochs=30,
    model_file='full_model.h5'
)
```

### Training Options

#### Command-Line Flags

**Basic Options:**
- `--n-images N` - Number of simulated images (default: 500)
- `--image-size SIZE` - Image size in pixels (default: 1024)
- `--epochs N` - Training epochs (default: 30)
- `--batch-size N` - Batch size (default: 64)
- `--val-split FRAC` - Validation split (default: 0.15)
- `--output FILE` - Output model file (default: realbogus_model.h5)

**Source Parameters:**
- `--n-stars-min N` / `--n-stars-max N` - Stars per image (default: 30-80)
- `--n-galaxies-min N` / `--n-galaxies-max N` - Galaxies per image (default: 10-30)
- `--fwhm-min F` / `--fwhm-max F` - FWHM range in pixels (default: 2.0-6.0)

**Artifact Parameters:**
- `--n-cr-min N` / `--n-cr-max N` - Cosmic rays (default: 5-20)
- `--n-hot-min N` / `--n-hot-max N` - Hot pixels (default: 10-30)
- `--bg-min N` / `--bg-max N` - Background level (default: 100-10000)

**Training Control:**
- `--include-galaxies` - Include galaxies as real sources
- `--no-class-weight` - Disable balanced class weighting
- `--no-augment` - Disable data augmentation
- `--plot` - Generate training history plot
- `--save-data` - Save training data to .npz file
- `--data FILE` - Load training data from .npz file

### Advanced Training

**Custom Simulation Parameters:**
```bash
python train_realbogus.py train \
    --n-images 1000 \
    --image-size 2048 \
    --n-stars-min 50 --n-stars-max 150 \
    --n-galaxies-min 10 --n-galaxies-max 40 \
    --fwhm-min 1.5 --fwhm-max 8.0 \
    --n-cr-min 10 --n-cr-max 30 \
    --epochs 50 \
    --batch-size 128 \
    --output production_model.h5 \
    --plot \
    --save-data
```

**Reusing Training Data:**
```bash
# Generate and save data
python train_realbogus.py train \
    --n-images 1000 \
    --output model_v1.h5 \
    --save-data

# Reuse for different architectures
python train_realbogus.py train \
    --data realbogus_model.npz \
    --epochs 50 \
    --output model_v2.h5
```

---

## Testing

### Single Image Testing

Test the classifier on a single image with comprehensive visualization.

**Command-Line:**
```bash
# Test on real FITS image
python train_realbogus.py test \
    --model model.h5 \
    --image observation.fits \
    --output test_results.png

# Test interactively
python train_realbogus.py test \
    --model model.h5 \
    --image observation.fits \
    --interactive

# Test on simulated image
python train_realbogus.py test \
    --model model.h5 \
    --fwhm 3.5 \
    --output test_sim.png
```

**Python:**
```python
from stdpipe import realbogus, photometry, simulation

# Load model
model = realbogus.load_realbogus_model('model.h5')

# Simulate or load image
sim = simulation.simulate_image(width=1024, height=1024)
image = sim['image']

# Detect and classify
objects = photometry.get_objects_sep(image, thresh=3.0)
real_objects = realbogus.classify_realbogus(
    objects, image, model=model, threshold=0.5
)
```

### Test Options

- `--model FILE` - Path to trained model (required)
- `--image FILE` - FITS image to test (if omitted, simulates one)
- `--image-size SIZE` - Simulated image size (default: 1024)
- `--fwhm F` - FWHM for detection/simulation (default: 3.5)
- `--threshold T` - Detection threshold in sigma (default: 3.0)
- `--rb-threshold T` - Real-bogus threshold (default: 0.5)
- `--interactive` - Show interactive plot window
- `--output FILE` - Output plot file (default: test_result.png)

### Test Output

The test command produces a 9-panel visualization:

1. **All Detections** - All detected objects
2. **Real Sources** - Objects classified as real
3. **Bogus Sources** - Objects classified as bogus
4. **Score Distribution** - Histogram of scores
5. **Score vs Flux** - Correlation analysis
6. **Score vs FWHM** - FWHM dependency
7. **Truth Comparison** - If simulated data
8. **Cumulative Distribution** - CDF of scores
9. **Statistics Summary** - Detailed metrics

---

## Evaluation

### Multi-Image Evaluation

Evaluate classifier performance on multiple test images with comprehensive metrics.

**Command-Line:**
```bash
# Basic evaluation
python train_realbogus.py evaluate \
    --model model.h5 \
    --n-images 100 \
    --output evaluation/

# Custom parameters
python train_realbogus.py evaluate \
    --model model.h5 \
    --n-images 200 \
    --rb-threshold 0.6 \
    --fwhm-min 2.0 --fwhm-max 7.0 \
    --match-radius 2.5 \
    --output eval_strict/
```

### Evaluation Options

**Basic Options:**
- `--model FILE` - Path to trained model (required)
- `--n-images N` - Number of test images (default: 50)
- `--image-size SIZE` - Test image size (default: 1024)
- `--fwhm-min F` / `--fwhm-max F` - FWHM range (default: 2.0-6.0)
- `--threshold T` - Detection threshold in sigma (default: 3.0)
- `--rb-threshold T` - Real-bogus threshold (default: 0.5)
- `--match-radius R` - Truth matching radius in pixels (default: 3.0)
- `--output DIR` - Output directory (default: evaluation/)

**Simulation Parameters:**
- `--n-stars-min N` / `--n-stars-max N` - Stars per image (default: 30-80)
- `--n-galaxies-min N` / `--n-galaxies-max N` - Galaxies per image (default: 10-30)
- `--n-cr-min N` / `--n-cr-max N` - Cosmic rays (default: 5-20)
- `--n-hot-min N` / `--n-hot-max N` - Hot pixels (default: 10-30)
- `--bg-min N` / `--bg-max N` - Background level (default: 100-10000)

**Interactive Visualization:**
- `--interactive` - Show interactive visualization of sample images
- `--n-interactive N` - Number of images to show (default: 3)

### Evaluation Outputs

1. **metrics.json** - Complete metrics in JSON format:
   ```json
   {
     "precision": 0.923,
     "precision_std": 0.045,
     "recall": 0.887,
     "recall_std": 0.052,
     "f1": 0.904,
     "f1_std": 0.038,
     "accuracy": 0.914,
     "accuracy_std": 0.041,
     "tp": 1234, "fp": 102, "tn": 876, "fn": 156,
     "n_images": 100
   }
   ```

2. **evaluation_results.png** - 6-panel diagnostic plot:
   - Metrics across images (precision, recall, F1)
   - Score distributions by true class
   - Confusion matrix heatmap
   - Metric distributions (box plots)
   - Cumulative score distributions

3. **interactive_sample_N.png** - If `--interactive` is used:
   - 3-panel visualization for each sample
   - All detections, real vs bogus, score distribution

### Custom Simulation Parameters

Control test image properties to evaluate robustness:

**Crowded Fields:**
```bash
python train_realbogus.py evaluate \
    --model model.h5 \
    --n-images 100 \
    --n-stars-min 150 --n-stars-max 250 \
    --n-galaxies-min 50 --n-galaxies-max 100 \
    --output eval_crowded/
```

**Sparse Fields with High Artifacts:**
```bash
python train_realbogus.py evaluate \
    --model model.h5 \
    --n-images 100 \
    --n-stars-min 10 --n-stars-max 30 \
    --n-cr-min 30 --n-cr-max 60 \
    --n-hot-min 50 --n-hot-max 100 \
    --output eval_artifacts/
```

**Faint Fields (Low Background):**
```bash
python train_realbogus.py evaluate \
    --model model.h5 \
    --n-images 100 \
    --bg-min 50 --bg-max 200 \
    --output eval_faint/
```

### Interactive Evaluation

Display sample images with real/bogus markings:

```bash
python train_realbogus.py evaluate \
    --model model.h5 \
    --n-images 50 \
    --interactive \
    --n-interactive 5 \
    --output eval_interactive/
```

**Interactive visualization shows:**
- **Panel 1**: All detections (yellow) + true stars (cyan crosses)
- **Panel 2**: Real sources (green) vs Bogus (red)
- **Panel 3**: Score distribution with threshold line

Close each window to proceed to the next sample.

### Performance Metrics

- **Precision**: Fraction of classified real sources that are truly real
- **Recall**: Fraction of true real sources detected as real
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall fraction of correct classifications
- **Confusion Matrix**: TP, FP, TN, FN counts

---

## Use Cases

### Case 1: Transient Detection (Default)

Stars-only classifier rejects stationary galaxies:

```bash
python train_realbogus.py train \
    --n-images 1000 \
    --n-galaxies-min 20 --n-galaxies-max 50 \
    --fwhm-min 2.5 --fwhm-max 5.0 \
    --epochs 50 \
    --output transient_model.h5
```

### Case 2: Poor Seeing Conditions

Train for wide FWHM range:

```bash
python train_realbogus.py train \
    --n-images 800 \
    --fwhm-min 4.0 --fwhm-max 10.0 \
    --epochs 40 \
    --output poor_seeing_model.h5
```

### Case 3: High Artifact Rate

Train with elevated cosmic ray and hot pixel rates:

```bash
python train_realbogus.py train \
    --n-images 600 \
    --n-cr-min 15 --n-cr-max 40 \
    --n-hot-min 20 --n-hot-max 60 \
    --epochs 35 \
    --output high_artifact_model.h5
```

### Case 4: Galaxy Surveys

Include galaxies as real sources:

```bash
python train_realbogus.py train \
    --include-galaxies \
    --n-images 1000 \
    --n-galaxies-min 30 --n-galaxies-max 80 \
    --epochs 50 \
    --output galaxy_survey_model.h5
```

### Case 5: Robustness Testing

Evaluate under different conditions:

```bash
# Crowded fields
python train_realbogus.py evaluate \
    --model model.h5 \
    --n-images 100 \
    --n-stars-min 150 --n-stars-max 250 \
    --output eval_crowded/

# High artifacts
python train_realbogus.py evaluate \
    --model model.h5 \
    --n-cr-min 30 --n-cr-max 60 \
    --n-hot-min 50 --n-hot-max 100 \
    --output eval_artifacts/
```

---

## Tips and Best Practices

### Training

1. **Use enough images**: 500-1000 for production models
2. **Match your conditions**: Set FWHM, background ranges to match real data
3. **Include sufficient artifacts**: Ensures good rejection of spurious detections
4. **Save training data**: Reuse for hyperparameter tuning
5. **Plot training curves**: Check for overfitting/underfitting
6. **Choose appropriate mode**: Stars-only for transients, full for galaxy surveys

### Testing

1. **Test on real data**: Simulations don't capture all real artifacts
2. **Use interactive mode**: Inspect results visually
3. **Try different thresholds**: Find optimal balance for your science case
4. **Check edge cases**: Test on crowded fields, bright stars, etc.

### Evaluation

1. **Use many test images**: 100+ for reliable statistics
2. **Vary conditions**: Test across FWHM range, backgrounds, source densities
3. **Match test to deployment**: Use similar source counts and artifact rates
4. **Test edge cases**: Crowded fields, sparse fields, high artifacts
5. **Check confusion matrix**: Understand failure modes
6. **Compare thresholds**: Choose based on precision/recall trade-off
7. **Use interactive mode**: Visually inspect classification results

### Threshold Selection

- **Low threshold (0.3-0.4)**: High completeness, more false positives
- **Medium threshold (0.5-0.6)**: Balanced (recommended default)
- **High threshold (0.7-0.9)**: High purity, may miss faint real sources

### Troubleshooting

**Low Precision** (many false positives):
- Increase `--rb-threshold` (e.g., 0.7 instead of 0.5)
- Train with more artifact examples
- Increase training epochs

**Low Recall** (missing real sources):
- Decrease `--rb-threshold` (e.g., 0.3 instead of 0.5)
- Train with more real source examples
- Check FWHM range matches your data

**Poor Performance on Real Data**:
- Match simulation parameters to real conditions
- Add more diverse artifacts to training
- Consider fine-tuning on labeled real data

---

## Advanced Features

### Interactive Visualization

#### During Evaluation

Display sample images interactively to visually inspect classification:

```bash
python train_realbogus.py evaluate \
    --model model.h5 \
    --n-images 100 \
    --interactive \
    --n-interactive 5 \
    --output eval_results/
```

**Features:**
- 3-panel figure for each sample
- All detections with truth overlays
- Real vs bogus separation
- Score histograms
- Interactive matplotlib windows
- Saved PNG files for later reference

#### During Testing

```bash
python train_realbogus.py test \
    --model model.h5 \
    --image observation.fits \
    --interactive
```

### Custom Simulation Parameters

Full control over test image properties:

**All Parameters Available:**
- Source counts (stars, galaxies)
- Artifact counts (cosmic rays, hot pixels)
- Background levels (log-uniform sampling)
- FWHM ranges
- Image sizes

**Example - Stress Testing:**
```bash
python train_realbogus.py evaluate \
    --model model.h5 \
    --n-stars-min 300 --n-stars-max 400 \
    --n-galaxies-min 100 --n-galaxies-max 150 \
    --n-cr-min 50 --n-cr-max 100 \
    --output eval_stress/
```

### Parameter Consistency

Evaluation defaults match training defaults:

| Parameter | Train | Evaluate | Match |
|-----------|-------|----------|-------|
| n-stars | 30-80 | 30-80 | ✓ |
| n-galaxies | 10-30 | 10-30 | ✓ |
| n-cr | 5-20 | 5-20 | ✓ |
| n-hot | 10-30 | 10-30 | ✓ |
| bg | 100-10000 | 100-10000 | ✓ |
| fwhm | 2.0-6.0 | 2.0-6.0 | ✓ |

### Data Augmentation

Training data is automatically augmented with:
- **Rotations**: 90°, 180°, 270°
- **Flips**: Horizontal and vertical
- **Result**: ~4× more training examples

Disable with `--no-augment` flag if needed.

### Class Balancing

Training automatically applies balanced class weighting to handle imbalanced datasets (more bogus than real sources). Disable with `--no-class-weight` if needed.

---

## Implementation History

### Stars-Only Default Change

**Rationale:**

Changed default behavior to treat only stars as real sources (galaxies as bogus) to better serve the most common use cases:
- Transient detection
- Variable star surveys
- Astrometry
- Crowded field photometry

**Changes Made:**

1. **Core Modules** (`stdpipe/simulation.py`, `stdpipe/realbogus.py`):
   - Changed `real_source_types` default from `['star', 'galaxy']` to `['star']`
   - Updated docstrings

2. **CLI Tool** (`examples/train_realbogus.py`):
   - Changed `--stars-only` flag to `--include-galaxies`
   - Inverted logic: now opts IN to including galaxies

3. **Documentation**:
   - Updated all examples and guides
   - Emphasized stars-only as default

**Migration:**

Users who want the old behavior must now explicitly specify:
```python
real_source_types=['star', 'galaxy']
```

Or use the `--include-galaxies` flag in CLI.

### Interactive Evaluation Feature

**Added:** Interactive visualization capability to evaluation command

**New Flags:**
- `--interactive` - Enable interactive display
- `--n-interactive N` - Number of samples to show (default: 3)

**Features:**
- 3-panel matplotlib windows
- Visual inspection of classification
- Saved PNG files
- Works with all evaluation options

### Simulation Parameters in Evaluation

**Added:** Full control over simulation parameters during evaluation

**New Flags:**
- Source counts (stars, galaxies)
- Artifact counts (cosmic rays, hot pixels)
- Background levels

**Benefits:**
- Robustness testing
- Edge case analysis
- Deployment condition matching
- Parameter sensitivity analysis

### JSON Serialization Fix

**Fixed:** TypeError when saving evaluation metrics with NumPy types

**Solution:** Convert NumPy types to native Python types before JSON serialization

---

## Complete Workflow Example

### 1. Train a Classifier

```bash
# Train stars-only classifier (default)
python train_realbogus.py train \
    --n-images 500 \
    --epochs 30 \
    --fwhm-min 2.0 --fwhm-max 6.0 \
    --output stars_model.h5 \
    --plot \
    --save-data
```

### 2. Test on Single Image

```bash
# Test interactively on real observation
python train_realbogus.py test \
    --model stars_model.h5 \
    --image /path/to/observation.fits \
    --rb-threshold 0.5 \
    --output test_result.png \
    --interactive
```

### 3. Evaluate Performance

```bash
# Comprehensive evaluation
python train_realbogus.py evaluate \
    --model stars_model.h5 \
    --n-images 100 \
    --output stars_eval/

# Check metrics
cat stars_eval/metrics.json

# Interactive evaluation
python train_realbogus.py evaluate \
    --model stars_model.h5 \
    --n-images 50 \
    --interactive \
    --n-interactive 5 \
    --output stars_eval_interactive/
```

### 4. Compare Thresholds

```bash
# Evaluate at different thresholds
for thresh in 0.3 0.5 0.7 0.9; do
    python train_realbogus.py evaluate \
        --model stars_model.h5 \
        --n-images 50 \
        --rb-threshold $thresh \
        --output eval_thresh_${thresh}/
done
```

### 5. Use in Python Pipeline

```python
from stdpipe import realbogus, photometry
from astropy.io import fits

# Load model
model = realbogus.load_realbogus_model('stars_model.h5')

# Process observations
for filename in observation_files:
    # Load image
    image = fits.getdata(filename)
    
    # Detect objects
    objects = photometry.get_objects_sep(image, thresh=3.0)
    
    # Classify
    real_objects = realbogus.classify_realbogus(
        objects, image,
        model=model,
        threshold=0.5
    )
    
    print(f"{filename}: {len(real_objects)} real sources")
```

---

## See Also

- **API Documentation**: `help(realbogus.train_realbogus_classifier)`
- **Python Examples**: `examples/realbogus_example.py`
- **Simulation Module**: `stdpipe.simulation`
- **Photometry Module**: `stdpipe.photometry`

---

## Citation

If you use the real-bogus classifier in your research, please cite the STDPipe package.

---

*Last updated: January 2026*
