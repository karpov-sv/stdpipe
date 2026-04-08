"""
Real-Bogus Classifier for Astronomical Object Detection

This module provides a CNN-based classifier to distinguish real astronomical sources
(stars, galaxies) from artifacts (cosmic rays, hot pixels, satellite trails) in
detected object catalogs.

Key Features:
- FWHM-invariant: Hybrid downscaling to canonical PSF size (no auxiliary FWHM input)
- Brightness-invariant: Peak normalization generalizes to any flux level
- Pure morphology: Classification based solely on source shape from 2-channel images
- 2-channel input: background-subtracted (linear), asinh-scaled (dynamic range compression)
- Lightweight 5-layer CNN (~100k parameters)
- Batch processing for efficient inference
- Optional TensorFlow dependency

Usage:
    from stdpipe import photometry, realbogus

    # Detect objects
    obj = photometry.get_objects_sep(image, thresh=3.0)

    # Classify and filter
    obj_clean = realbogus.classify_realbogus(obj, image, threshold=0.5)

    # Or add scores without filtering
    obj = realbogus.classify_realbogus(obj, image, add_score=True, flag_bogus=False)
    print(obj['rb_score'])

Author: STDPipe Contributors
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import os
import warnings
from astropy.table import Column

# Conditional TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    warnings.warn(
        "TensorFlow not found. Real-bogus classifier will not be available. "
        "Install with: pip install stdpipe[ml]",
        ImportWarning,
    )


# Model architecture parameters
DEFAULT_MODEL_DIR = os.path.expanduser('~/.stdpipe/models')
DEFAULT_MODEL_NAME = 'realbogus_default.h5'
TARGET_FWHM = 3.0  # Canonical FWHM for downscaling normalization
DEFAULT_ASINH_SOFTENING_SIGMA = 3.0  # Asinh softening in units of background sigma


def _check_tensorflow():
    """Check if TensorFlow is available, raise error if not."""
    if not HAS_TENSORFLOW:
        raise ImportError(
            "TensorFlow is required for real-bogus classification. "
            "Install with: pip install stdpipe[ml] or pip install tensorflow>=2.10"
        )


def create_realbogus_model(
    input_shape=(31, 31, 2), filters=(32, 64, 128), dense_units=64, dropout_rate=0.5
):
    """
    Create CNN architecture for real-bogus classification.

    Architecture:
        - 3-5 convolutional layers with batch normalization
        - Global average pooling (handles variable input sizes)
        - Dense layer with dropout
        - Sigmoid output (binary classification)

    Design Philosophy:
        - FWHM-invariant: Images downscaled to canonical FWHM, no auxiliary FWHM input needed
        - Brightness-invariant: Peak normalization allows generalization to any flux level
        - Pure morphology: Classification based solely on source shape

    Input Channels:
        - Channel 0: Background-subtracted (linear scale), peak-normalized
        - Channel 1: Asinh-scaled background-subtracted, peak-normalized

    Parameters
    ----------
    input_shape : tuple, optional
        Input shape (height, width, channels). Default: (31, 31, 2)
        Height/width can be None for variable-size inputs.
    filters : tuple, optional
        Number of filters in each conv layer. Default: (32, 64, 128)
    dense_units : int, optional
        Units in dense layer. Default: 64
    dropout_rate : float, optional
        Dropout rate for regularization. Default: 0.5

    Returns
    -------
    model : keras.Model
        Compiled Keras model ready for training
    """
    _check_tensorflow()

    # Main image input (2 channels: background-subtracted linear, asinh-scaled)
    image_input = keras.Input(shape=input_shape, name='image_input')

    # Convolutional feature extraction
    x = image_input
    for i, n_filters in enumerate(filters):
        x = layers.Conv2D(
            n_filters, (3, 3), activation='relu', padding='same', name=f'conv{i + 1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn{i + 1}')(x)
        x = layers.MaxPooling2D((2, 2), name=f'pool{i + 1}')(x)

    # Global pooling to handle variable input sizes
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)

    # Dense layers (no auxiliary inputs - pure image-based classification)
    x = layers.Dense(dense_units, activation='relu', name='dense1')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)

    # Output layer (sigmoid for binary classification)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    # Create and compile model
    model = keras.Model(inputs=image_input, outputs=output, name='realbogus_classifier')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')],
    )

    return model


def _downscale_cutout(cutout, scale_factor, mode='mean'):
    """
    Downscale cutout by integer factor using block averaging.

    Parameters
    ----------
    cutout : ndarray
        2D image cutout
    scale_factor : int or float
        Downscaling factor (converted to integer, must be >= 1).
        Typically passed as round(fwhm/target_fwhm) for robust hot pixel suppression.
    mode : str, optional
        Downscaling mode: 'mean' or 'median'. Default: 'mean'

    Returns
    -------
    downscaled : ndarray
        Downscaled cutout
    """
    if scale_factor <= 1:
        return cutout

    scale_factor = int(scale_factor)
    h, w = cutout.shape

    # Trim to multiple of scale_factor
    new_h = (h // scale_factor) * scale_factor
    new_w = (w // scale_factor) * scale_factor
    cutout_trimmed = cutout[:new_h, :new_w]

    # Reshape and aggregate
    reshaped = cutout_trimmed.reshape(
        new_h // scale_factor, scale_factor, new_w // scale_factor, scale_factor
    )

    if mode == 'median':
        downscaled = np.median(reshaped, axis=(1, 3))
    else:  # mean
        downscaled = np.mean(reshaped, axis=(1, 3))

    return downscaled


def _infer_cutout_radius_from_model(model, default_radius=15, log=None):
    """Infer cutout radius from model input shape."""
    if log is None:
        log = lambda *args, **kwargs: None

    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0] if shape else None

    if shape is None or len(shape) < 4:
        log(
            f"Model input shape unavailable; using default cutout size {2 * default_radius + 1} "
            f"(radius {default_radius})"
        )
        return default_radius

    height, width = shape[1], shape[2]
    channels = shape[3]

    if height is None or width is None:
        log(
            f"Model input shape is dynamic; using default cutout size {2 * default_radius + 1} "
            f"(radius {default_radius})"
        )
        return default_radius

    if height != width:
        raise ValueError(
            f"Model input shape must be square for cutout extraction (got {height}x{width})."
        )
    if height % 2 == 0:
        raise ValueError(f"Model input size must be odd for symmetric cutouts (got {height}).")

    radius = int((height - 1) // 2)
    log(f"Using cutout size {height}x{width} (radius {radius}) from model input shape")

    if channels is not None and channels != 2:
        log(f"Warning: model expects {channels} channels; realbogus generates 2-channel cutouts")

    return radius


def _upscale_cutout(cutout, scale_factor):
    """
    Upscale cutout by integer factor using pixel replication.

    Parameters
    ----------
    cutout : ndarray
        2D image cutout
    scale_factor : int or float
        Upscaling factor (converted to integer, must be >= 1).
        Each pixel is replicated into scale_factor x scale_factor block.

    Returns
    -------
    upscaled : ndarray
        Upscaled cutout
    """
    if scale_factor <= 1:
        return cutout

    scale_factor = int(scale_factor)

    # Use numpy repeat for integer upscaling
    # Repeat along each axis
    upscaled = np.repeat(cutout, scale_factor, axis=0)
    upscaled = np.repeat(upscaled, scale_factor, axis=1)

    return upscaled


def _pad_to_size(cutout, target_size, mode='edge'):
    """
    Pad cutout to target size (centered).

    Parameters
    ----------
    cutout : ndarray
        2D image cutout
    target_size : int
        Target size (square)
    mode : str, optional
        Padding mode for np.pad. Default: 'edge'

    Returns
    -------
    padded : ndarray
        Padded cutout (target_size x target_size)
    """
    h, w = cutout.shape

    # Crop dimensions that exceed target
    if h > target_size:
        start_h = (h - target_size) // 2
        cutout = cutout[start_h : start_h + target_size, :]
        h = target_size
    if w > target_size:
        start_w = (w - target_size) // 2
        cutout = cutout[:, start_w : start_w + target_size]
        w = target_size

    # Pad dimensions that are smaller than target
    pad_h = target_size - h
    pad_w = target_size - w
    if pad_h > 0 or pad_w > 0:
        cutout = np.pad(
            cutout, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), mode=mode
        )

    return cutout


def preprocess_cutout(
    cutout_sci,
    cutout_bg=None,
    cutout_err=None,
    fwhm=None,
    target_fwhm=TARGET_FWHM,
    target_size=31,
    downscale_threshold=1.5,
    normalize=True,
    asinh_softening=None,
):
    """
    Preprocess cutout for CNN input.

    Steps:
        1. Optional scaling (downscale or upscale) to canonical FWHM
        2. Create 2-channel input (background-subtracted linear, asinh-scaled)
        3. Peak normalization (each channel normalized by its own peak value)
        4. Pad/crop to target size

    FWHM Scaling Strategy (Symmetric):
        - Downscaling (FWHM > target_fwhm × threshold): Integer block averaging
        - No scaling (target_fwhm / threshold ≤ FWHM ≤ target_fwhm × threshold): Keep as-is
        - Upscaling (FWHM < target_fwhm / threshold): Integer pixel replication

        Default: target_fwhm=3.0, threshold=1.5
        → Downscale if FWHM > 4.5, upscale if FWHM < 2.0, else unchanged

        This ensures all PSFs normalized to approximately the same size regardless of
        sharpness, eliminating FWHM as a confounding variable.

    Channel Design:
        - Channel 0: Background-subtracted (linear scale), peak-normalized
        - Channel 1: Asinh-scaled background-subtracted, peak-normalized

        Peak normalization makes the representation brightness-invariant: all sources
        (faint to extremely bright) are scaled to [-1, 1] range based on their peak
        value. This allows the CNN to learn pure morphological features that generalize
        to ANY brightness level, including sources far brighter than the training set.

        The asinh channel complements the linear channel by providing compressed
        dynamic range information useful for distinguishing extended vs. compact sources.

    Parameters
    ----------
    cutout_sci : ndarray
        Science image cutout (assumed to be background-subtracted if cutout_bg is None)
    cutout_bg : ndarray, optional
        Background cutout (or scalar value). If None, estimated from cutout edges.
    cutout_err : ndarray or float, optional
        Error/noise cutout (or scalar value). Used to estimate the noise level
        (sigma) for asinh softening. Only the median value is used.
    fwhm : float, optional
        Image FWHM in pixels. If provided, cutout will be downscaled to target_fwhm.
    target_fwhm : float, optional
        Target FWHM for downscaling normalization. Default: 3.0
    target_size : int, optional
        Target cutout size (square). Default: 31
    downscale_threshold : float, optional
        Only downscale if fwhm/target_fwhm > threshold. Default: 1.5
    normalize : bool, optional
        Apply peak normalization to each channel (scales to [-1, 1] range).
        Default: True. This makes the representation brightness-invariant.
    asinh_softening : float, optional
        Asinh softening in units of background sigma. If None, uses
        DEFAULT_ASINH_SOFTENING_SIGMA. Actual softening is
        (asinh_softening * sigma), where sigma is estimated from cutout_err.

    Returns
    -------
    preprocessed : ndarray
        Preprocessed cutout (target_size, target_size, 2)
    scale_factor : float
        Applied scale factor (for diagnostics)
    """
    # Handle background
    if cutout_bg is None:
        # Estimate from edges
        edge_pixels = np.concatenate(
            [cutout_sci[0, :], cutout_sci[-1, :], cutout_sci[:, 0], cutout_sci[:, -1]]
        )
        cutout_bg = np.median(edge_pixels)

    if np.isscalar(cutout_bg):
        cutout_bg = np.full_like(cutout_sci, cutout_bg)

    # Estimate noise sigma for asinh softening (only the median is needed,
    # so compute before any scaling to avoid unnecessary array operations)
    if cutout_err is None:
        from astropy.stats import mad_std

        sigma = float(mad_std(cutout_sci))
    elif np.isscalar(cutout_err):
        sigma = float(cutout_err)
    else:
        sigma = float(np.nanmedian(cutout_err))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    # Scaling to canonical FWHM (both downscale and upscale)
    scale_factor = 1.0
    if fwhm is not None and fwhm > 0:
        scale_factor = fwhm / target_fwhm

        # Downscale if PSF too large
        if scale_factor > downscale_threshold:
            factor = round(scale_factor)
            cutout_sci = _downscale_cutout(cutout_sci, factor)
            cutout_bg = _downscale_cutout(cutout_bg, factor)

        # Upscale if PSF too sharp (symmetric with downscaling)
        elif scale_factor < 1.0 / downscale_threshold:
            factor = round(1.0 / scale_factor)
            cutout_sci = _upscale_cutout(cutout_sci, factor)
            cutout_bg = _upscale_cutout(cutout_bg, factor)

    # Create background-subtracted channel
    cutout_bgsub = cutout_sci - cutout_bg

    # Create asinh-scaled channel for dynamic range compression
    if asinh_softening is None:
        asinh_softening = DEFAULT_ASINH_SOFTENING_SIGMA

    softening = float(asinh_softening) * sigma
    if not np.isfinite(softening) or softening <= 0:
        softening = sigma

    # Asinh scaling: compresses high values, ~linear for low values
    cutout_asinh = np.arcsinh(cutout_bgsub / softening)

    # Stack channels: [background-subtracted linear, asinh-scaled]
    channels = [cutout_bgsub, cutout_asinh]

    # Peak normalization for brightness invariance
    if normalize:
        for i, ch in enumerate(channels):
            peak = np.max(np.abs(ch))
            if peak > 1e-10:
                channels[i] = ch / peak

    # Pad/crop to target size
    channels = [_pad_to_size(ch, target_size) for ch in channels]

    # Stack into 2-channel image
    preprocessed = np.stack(channels, axis=-1).astype(np.float32)

    return preprocessed, scale_factor


def extract_cutouts(
    obj,
    image,
    bg=None,
    err=None,
    mask=None,
    radius=15,
    fwhm=None,
    target_fwhm=TARGET_FWHM,
    asinh_softening=None,
    verbose=False,
):
    """
    Extract and preprocess cutouts for all objects.

    Parameters
    ----------
    obj : astropy.table.Table
        Object catalog with 'x' and 'y' columns
    image : ndarray
        Science image
    bg : ndarray or float, optional
        Background map or scalar value
    err : ndarray or float, optional
        Error/noise map or scalar value
    mask : ndarray, optional
        Boolean mask (True = masked)
    radius : int, optional
        Cutout radius in pixels. Default: 15 (31x31 cutouts)
    fwhm : float, optional
        Image FWHM. If None, estimated from object catalog.
    target_fwhm : float, optional
        Target FWHM for downscaling. Default: 3.0
    asinh_softening : float, optional
        Asinh softening in units of background sigma. If None, uses
        DEFAULT_ASINH_SOFTENING_SIGMA.
    verbose : bool, optional
        Print progress. Default: False

    Returns
    -------
    cutouts : ndarray
        Array of preprocessed cutouts (N, 2*radius+1, 2*radius+1, 2)
    valid_indices : ndarray
        Indices of successfully extracted cutouts
    """
    log = print if verbose else lambda *args, **kwargs: None

    # Estimate FWHM if not provided
    if fwhm is None:
        if 'fwhm' in obj.colnames:
            fwhm = np.median(obj['fwhm'])
        elif 'a' in obj.colnames:
            # Approximate from semi-major axis
            fwhm = np.median(obj['a']) * 2.35
        else:
            fwhm = 3.0  # Fallback
        log(f"Estimated FWHM: {fwhm:.2f} pixels")

    cutout_size = 2 * radius + 1
    cutouts = []
    valid_indices = []

    h, w = image.shape

    for i, row in enumerate(obj):
        x, y = row['x'], row['y']

        # Check bounds
        x_min = int(x - radius)
        x_max = int(x + radius + 1)
        y_min = int(y - radius)
        y_max = int(y + radius + 1)

        if x_min < 0 or x_max > w or y_min < 0 or y_max > h:
            log(f"Object {i} too close to edge, skipping")
            continue

        # Extract cutout
        cutout_sci = image[y_min:y_max, x_min:x_max].copy()

        # Extract background cutout
        if bg is not None:
            if np.isscalar(bg):
                cutout_bg = bg
            else:
                cutout_bg = bg[y_min:y_max, x_min:x_max].copy()
        else:
            cutout_bg = None

        # Extract error cutout
        if err is not None:
            if np.isscalar(err):
                cutout_err = err
            else:
                cutout_err = err[y_min:y_max, x_min:x_max].copy()
        else:
            cutout_err = None

        # Apply mask if provided
        if mask is not None:
            cutout_mask = mask[y_min:y_max, x_min:x_max]
            if np.sum(~cutout_mask) < 0.5 * cutout_sci.size:
                log(f"Object {i} too masked, skipping")
                continue
            # Set masked pixels to median
            if np.any(cutout_mask):
                median_val = np.median(cutout_sci[~cutout_mask])
                cutout_sci[cutout_mask] = median_val

        # Preprocess
        try:
            preprocessed, scale_factor = preprocess_cutout(
                cutout_sci,
                cutout_bg=cutout_bg,
                cutout_err=cutout_err,
                fwhm=fwhm,
                target_fwhm=target_fwhm,
                target_size=cutout_size,
                asinh_softening=asinh_softening,
            )

            cutouts.append(preprocessed)
            valid_indices.append(i)

        except Exception as e:
            log(f"Failed to preprocess object {i}: {e}")
            continue

    if len(cutouts) == 0:
        raise ValueError("No valid cutouts extracted")

    cutouts = np.array(cutouts, dtype=np.float32)
    valid_indices = np.array(valid_indices, dtype=int)

    log(f"Extracted {len(cutouts)} valid cutouts from {len(obj)} objects")

    return cutouts, valid_indices


def load_realbogus_model(model_file=None, verbose=False):
    """
    Load pre-trained real-bogus model.

    Parameters
    ----------
    model_file : str, optional
        Path to model file (.h5 or SavedModel directory).
        If None, loads default model from ~/.stdpipe/models/
    verbose : bool, optional
        Print loading information. Default: False

    Returns
    -------
    model : keras.Model
        Loaded Keras model
    """
    _check_tensorflow()

    log = print if verbose else lambda *args, **kwargs: None

    # Default model path
    if model_file is None:
        model_file = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME)

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Model file not found: {model_file}\n"
            "Train a model using train_realbogus_classifier() or specify model_file."
        )

    log(f"Loading model from {model_file}")
    model = keras.models.load_model(model_file)
    log(f"Model loaded successfully ({model.count_params()} parameters)")

    return model


def save_realbogus_model(model, model_file=None, verbose=False):
    """
    Save trained real-bogus model.

    Parameters
    ----------
    model : keras.Model
        Trained model
    model_file : str, optional
        Output path. If None, saves to ~/.stdpipe/models/realbogus_default.h5
    verbose : bool, optional
        Print saving information. Default: False
    """
    _check_tensorflow()

    log = print if verbose else lambda *args, **kwargs: None

    # Default model path
    if model_file is None:
        os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
        model_file = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME)

    # Ensure directory exists
    model_dir = os.path.dirname(model_file)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    log(f"Saving model to {model_file}")
    model.save(model_file)
    log("Model saved successfully")


def classify_realbogus(
    obj,
    image,
    model=None,
    model_file=None,
    bg=None,
    err=None,
    mask=None,
    fwhm=None,
    asinh_softening=None,
    threshold=0.5,
    add_score=True,
    flag_bogus=True,
    batch_size=128,
    verbose=False,
):
    """
    Classify detected objects as real or bogus using CNN.

    This is the main entry point for real-bogus classification.

    Parameters
    ----------
    obj : astropy.table.Table
        Object catalog with 'x' and 'y' columns (from photometry.get_objects_*)
    image : ndarray
        Science image
    model : keras.Model, optional
        Pre-loaded model. If None, loads from model_file.
    model_file : str, optional
        Path to model file. If None, uses default model.
    bg : ndarray or float, optional
        Background map or scalar value
    err : ndarray or float, optional
        Error/noise map or scalar value
    mask : ndarray, optional
        Boolean mask (True = masked pixels)
    cutout size : derived
        Cutout size is inferred from the model input shape. If the model has
        dynamic spatial dimensions, defaults to 31x31 (radius 15).
    fwhm : float, optional
        Image FWHM. If None, estimated from catalog.
    asinh_softening : float, optional
        Asinh softening in units of background sigma. If None, uses
        DEFAULT_ASINH_SOFTENING_SIGMA.
    threshold : float, optional
        Classification threshold (0-1). Objects with score > threshold are real.
        Default: 0.5
    add_score : bool, optional
        Add 'rb_score' column to output catalog. Default: True
    flag_bogus : bool, optional
        Set flags=0x1000 for bogus objects and filter them out. Default: True
    batch_size : int, optional
        Batch size for inference. Default: 128
    verbose : bool or callable, optional
        Print progress. Can be callable for custom logging. Default: False

    Returns
    -------
    obj_filtered : astropy.table.Table
        Filtered catalog with real sources only (if flag_bogus=True)
        or full catalog with 'rb_score' column (if flag_bogus=False)

    Examples
    --------
    >>> from stdpipe import photometry, realbogus
    >>> obj = photometry.get_objects_sep(image, thresh=3.0)
    >>> obj_clean = realbogus.classify_realbogus(obj, image)
    >>> print(f"Kept {len(obj_clean)}/{len(obj)} objects")
    """
    _check_tensorflow()

    # Handle verbose as callable
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Load model if not provided
    if model is None:
        model = load_realbogus_model(model_file=model_file, verbose=verbose)

    cutout_radius = _infer_cutout_radius_from_model(model, default_radius=15, log=log)

    log(f"Classifying {len(obj)} objects (threshold={threshold:.2f})")

    # Extract cutouts
    cutouts, valid_indices = extract_cutouts(
        obj,
        image,
        bg=bg,
        err=err,
        mask=mask,
        radius=cutout_radius,
        fwhm=fwhm,
        asinh_softening=asinh_softening,
        verbose=verbose,
    )

    # Batch inference
    log(f"Running inference on {len(cutouts)} cutouts (batch_size={batch_size})")
    predictions = model.predict(cutouts, batch_size=batch_size, verbose=1 if verbose else 0)

    # Flatten predictions
    scores = predictions.flatten()

    # Create full score array (NaN for invalid objects)
    full_scores = np.full(len(obj), np.nan, dtype=float)
    full_scores[valid_indices] = scores

    # Add score column if requested
    if add_score:
        if 'rb_score' in obj.colnames:
            obj['rb_score'] = full_scores
        else:
            obj.add_column(Column(full_scores, name='rb_score'))

    # Flag and filter bogus objects
    if flag_bogus:
        # Ensure flags column exists
        if 'flags' not in obj.colnames:
            obj.add_column(Column(np.zeros(len(obj), dtype=int), name='flags'))

        # Set bogus flag (0x1000)
        is_bogus = full_scores < threshold
        is_bogus[np.isnan(full_scores)] = True  # Flag invalid objects as bogus
        obj['flags'][is_bogus] |= 0x1000

        # Filter
        obj_filtered = obj[~is_bogus]
        log(
            f"Filtered: {len(obj_filtered)}/{len(obj)} objects retained "
            f"({100 * len(obj_filtered) / len(obj):.1f}%)"
        )

        return obj_filtered
    else:
        return obj


def train_realbogus_classifier(
    training_data=None,
    n_simulated=1000,
    image_size=(2048, 2048),
    fwhm_range=(1.5, 8.0),
    real_source_types=['star'],
    validation_split=0.15,
    model=None,
    model_file=None,
    epochs=50,
    batch_size=64,
    class_weight='balanced',
    callbacks=None,
    verbose=True,
):
    """
    Train real-bogus classifier on simulated or real data.

    Parameters
    ----------
    training_data : tuple or dict, optional
        Pre-generated training data: (X, y, fwhm_features) tuple or dict with 'X', 'y', 'fwhm' keys.
        If None, generates simulated data using simulation.generate_realbogus_training_data().
    n_simulated : int, optional
        Number of simulated images to generate (if training_data=None). Default: 1000
    image_size : tuple, optional
        Size of simulated images (width, height). Default: (2048, 2048)
    fwhm_range : tuple, optional
        Range of FWHM values for simulated images. Default: (1.5, 8.0)
    real_source_types : list, optional
        List of source types to consider 'real' (if training_data=None).
        Default: ['star'] treats only stars as real and galaxies as bogus.
        Use ['star', 'galaxy'] to train a classifier that treats both as real.
    validation_split : float, optional
        Fraction of data for validation. Default: 0.15
    model : keras.Model, optional
        Model to train. If None, creates new model.
    model_file : str, optional
        Path to save trained model. Default: ~/.stdpipe/models/realbogus_default.h5
    epochs : int, optional
        Training epochs. Default: 50
    batch_size : int, optional
        Batch size. Default: 64
    class_weight : str or dict, optional
        Class weights for imbalanced data. 'balanced' or dict {0: w0, 1: w1}.
        Default: 'balanced'
    callbacks : list, optional
        Keras callbacks (e.g., early stopping, checkpoints)
    verbose : bool, optional
        Print training progress. Default: True

    Returns
    -------
    model : keras.Model
        Trained model
    history : keras.callbacks.History
        Training history

    Examples
    --------
    >>> from stdpipe import realbogus
    >>> # Train on simulated data (stars and galaxies as real)
    >>> model, history = realbogus.train_realbogus_classifier(
    ...     n_simulated=500,
    ...     epochs=30,
    ...     verbose=True
    ... )
    >>> # Train stars-only classifier (galaxies as bogus)
    >>> model, history = realbogus.train_realbogus_classifier(
    ...     n_simulated=500,
    ...     real_source_types=['star'],
    ...     epochs=30,
    ...     verbose=True
    ... )
    >>> # Or use pre-generated data
    >>> data = realbogus.generate_training_data(...)
    >>> model, history = realbogus.train_realbogus_classifier(
    ...     training_data=(data['X'], data['y']),
    ...     epochs=30
    ... )
    """
    _check_tensorflow()

    log = print if verbose else lambda *args, **kwargs: None

    # Generate or load training data
    if training_data is None:
        log(f"Generating training data from {n_simulated} simulated images...")
        from . import simulation

        data = simulation.generate_realbogus_training_data(
            n_images=n_simulated,
            image_size=image_size,
            fwhm_range=fwhm_range,
            real_source_types=real_source_types,
            augment=True,
            verbose=verbose,
        )
        X = data['X']
        y = data['y']
    else:
        # Handle both tuple and dict formats
        if isinstance(training_data, dict):
            X = training_data['X']
            y = training_data['y']
        elif len(training_data) == 3:
            X, y, _fwhm_features = training_data  # Legacy 3-tuple format
        else:
            X, y = training_data

    log(f"Training data: {len(X)} samples, {np.sum(y)} real, {len(y) - np.sum(y)} bogus")

    # Create model if not provided
    if model is None:
        input_shape = X.shape[1:]  # (height, width, channels)
        model = create_realbogus_model(input_shape=input_shape)

    model.summary(print_fn=log)

    # Calculate class weights
    if class_weight == 'balanced':
        n_real = np.sum(y)
        n_bogus = len(y) - n_real
        if n_real == 0 or n_bogus == 0:
            log("Warning: Only one class present in training data; disabling class weighting")
            class_weight = None
        else:
            class_weight = {0: len(y) / (2 * n_bogus), 1: len(y) / (2 * n_real)}
            log(f"Class weights: {class_weight}")

    # Default callbacks
    if callbacks is None:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

    # Train model (pure image-based, no FWHM auxiliary input)
    log("Starting training...")
    history = model.fit(
        X,
        y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1 if verbose else 0,
    )

    log("Training complete")

    # Save model only when explicitly requested
    if model_file is not None:
        save_realbogus_model(model, model_file=model_file, verbose=verbose)

    return model, history
