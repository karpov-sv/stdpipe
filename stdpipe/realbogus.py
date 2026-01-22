"""
Real-Bogus Classifier for Astronomical Object Detection

This module provides a CNN-based classifier to distinguish real astronomical sources
(stars, galaxies) from artifacts (cosmic rays, hot pixels, satellite trails) in
detected object catalogs.

Key Features:
- FWHM-invariant design using hybrid downscaling approach
- 3-channel input: science, background-subtracted, SNR map
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
from astropy.table import Table, Column
from astropy.stats import mad_std

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
        ImportWarning
    )


# Model architecture parameters
DEFAULT_MODEL_DIR = os.path.expanduser('~/.stdpipe/models')
DEFAULT_MODEL_NAME = 'realbogus_default.h5'
TARGET_FWHM = 3.0  # Canonical FWHM for downscaling normalization


def _check_tensorflow():
    """Check if TensorFlow is available, raise error if not."""
    if not HAS_TENSORFLOW:
        raise ImportError(
            "TensorFlow is required for real-bogus classification. "
            "Install with: pip install stdpipe[ml] or pip install tensorflow>=2.10"
        )


def create_realbogus_model(
    input_shape=(31, 31, 3),
    use_fwhm_feature=True,
    filters=(32, 64, 128),
    dense_units=64,
    dropout_rate=0.5
):
    """
    Create CNN architecture for real-bogus classification.

    Architecture:
        - 3-5 convolutional layers with batch normalization
        - Global average pooling (handles variable input sizes)
        - Optional FWHM auxiliary input
        - Dense layer with dropout
        - Sigmoid output (binary classification)

    Parameters
    ----------
    input_shape : tuple, optional
        Input shape (height, width, channels). Default: (31, 31, 3)
        Height/width can be None for variable-size inputs.
    use_fwhm_feature : bool, optional
        Include FWHM as auxiliary input feature. Default: True
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

    # Main image input (3 channels: science, bgsub, snr)
    image_input = keras.Input(shape=input_shape, name='image_input')

    # Convolutional feature extraction
    x = image_input
    for i, n_filters in enumerate(filters):
        x = layers.Conv2D(
            n_filters, (3, 3),
            activation='relu',
            padding='same',
            name=f'conv{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn{i+1}')(x)
        x = layers.MaxPooling2D((2, 2), name=f'pool{i+1}')(x)

    # Global pooling to handle variable input sizes
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)

    # Optional FWHM auxiliary input
    if use_fwhm_feature:
        fwhm_input = keras.Input(shape=(1,), name='fwhm_input')
        x = layers.Concatenate(name='concat')([x, fwhm_input])
        inputs = [image_input, fwhm_input]
    else:
        inputs = image_input

    # Dense layers
    x = layers.Dense(dense_units, activation='relu', name='dense1')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)

    # Output layer (sigmoid for binary classification)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=output, name='realbogus_classifier')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    return model


def _normalize_cutout(cutout, method='robust'):
    """
    Normalize cutout using z-score normalization.

    Parameters
    ----------
    cutout : ndarray
        2D image cutout
    method : str, optional
        Normalization method: 'robust' (median/MAD) or 'standard' (mean/std)
        Default: 'robust'

    Returns
    -------
    normalized : ndarray
        Normalized cutout (mean=0, std=1)
    """
    if method == 'robust':
        center = np.median(cutout)
        scale = mad_std(cutout)
    else:
        center = np.mean(cutout)
        scale = np.std(cutout)

    # Avoid division by zero
    if scale < 1e-10:
        scale = 1.0

    return (cutout - center) / scale


def _downscale_cutout(cutout, scale_factor, mode='mean'):
    """
    Downscale cutout by integer factor using block averaging.

    Parameters
    ----------
    cutout : ndarray
        2D image cutout
    scale_factor : int
        Downscaling factor (must be integer >= 1)
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
        new_h // scale_factor, scale_factor,
        new_w // scale_factor, scale_factor
    )

    if mode == 'median':
        downscaled = np.median(reshaped, axis=(1, 3))
    else:  # mean
        downscaled = np.mean(reshaped, axis=(1, 3))

    return downscaled


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
    if h >= target_size and w >= target_size:
        # Already large enough, crop center
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        return cutout[start_h:start_h+target_size, start_w:start_w+target_size]

    # Calculate padding
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = np.pad(
        cutout,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode=mode
    )

    return padded


def preprocess_cutout(
    cutout_sci,
    cutout_bg=None,
    cutout_err=None,
    fwhm=None,
    target_fwhm=TARGET_FWHM,
    target_size=31,
    downscale_threshold=1.5,
    normalize=True
):
    """
    Preprocess cutout for CNN input.

    Steps:
        1. Optional downscaling to canonical FWHM
        2. Create 3-channel input (science, bgsub, SNR)
        3. Normalize each channel
        4. Pad/crop to target size

    Parameters
    ----------
    cutout_sci : ndarray
        Science image cutout
    cutout_bg : ndarray, optional
        Background cutout (or scalar value). If None, estimated from cutout.
    cutout_err : ndarray, optional
        Error/noise cutout (or scalar value). If None, estimated from cutout.
    fwhm : float, optional
        Image FWHM in pixels. If provided, cutout will be downscaled to target_fwhm.
    target_fwhm : float, optional
        Target FWHM for downscaling normalization. Default: 3.0
    target_size : int, optional
        Target cutout size (square). Default: 31
    downscale_threshold : float, optional
        Only downscale if fwhm/target_fwhm > threshold. Default: 1.5
    normalize : bool, optional
        Apply z-score normalization. Default: True

    Returns
    -------
    preprocessed : ndarray
        Preprocessed cutout (target_size, target_size, 3)
    scale_factor : float
        Applied scale factor (for diagnostics)
    """
    # Handle background
    if cutout_bg is None:
        # Estimate from edges
        edge_pixels = np.concatenate([
            cutout_sci[0, :], cutout_sci[-1, :],
            cutout_sci[:, 0], cutout_sci[:, -1]
        ])
        cutout_bg = np.median(edge_pixels)

    if np.isscalar(cutout_bg):
        cutout_bg = np.full_like(cutout_sci, cutout_bg)

    # Handle error/noise
    if cutout_err is None:
        # Estimate from MAD
        cutout_err = mad_std(cutout_sci)

    if np.isscalar(cutout_err):
        cutout_err = np.full_like(cutout_sci, cutout_err)

    # Downscaling to canonical FWHM
    scale_factor = 1.0
    if fwhm is not None and fwhm > 0:
        scale_factor = fwhm / target_fwhm
        if scale_factor > downscale_threshold:
            cutout_sci = _downscale_cutout(cutout_sci, int(scale_factor))
            cutout_bg = _downscale_cutout(cutout_bg, int(scale_factor))
            cutout_err = _downscale_cutout(cutout_err, int(scale_factor))

    # Create background-subtracted channel
    cutout_bgsub = cutout_sci - cutout_bg

    # Create SNR channel (avoid division by zero)
    cutout_snr = np.where(
        cutout_err > 1e-10,
        cutout_bgsub / cutout_err,
        0.0
    )

    # Stack channels
    channels = [cutout_sci, cutout_bgsub, cutout_snr]

    # Normalize each channel
    if normalize:
        channels = [_normalize_cutout(ch) for ch in channels]

    # Pad/crop to target size
    channels = [_pad_to_size(ch, target_size) for ch in channels]

    # Stack into 3-channel image
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
    verbose=False
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
    verbose : bool, optional
        Print progress. Default: False

    Returns
    -------
    cutouts : ndarray
        Array of preprocessed cutouts (N, 2*radius+1, 2*radius+1, 3)
    fwhm_features : ndarray
        Array of normalized FWHM values (N, 1)
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

    # Prepare containers
    cutout_size = 2 * radius + 1
    cutouts = []
    fwhm_features = []
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
                target_size=cutout_size
            )

            # Normalize FWHM feature
            fwhm_norm = (fwhm / target_fwhm - 1.0) / 2.0  # ~[-0.5, 0.5] for FWHM=1-10

            cutouts.append(preprocessed)
            fwhm_features.append([fwhm_norm])
            valid_indices.append(i)

        except Exception as e:
            log(f"Failed to preprocess object {i}: {e}")
            continue

    if len(cutouts) == 0:
        raise ValueError("No valid cutouts extracted")

    cutouts = np.array(cutouts, dtype=np.float32)
    fwhm_features = np.array(fwhm_features, dtype=np.float32)
    valid_indices = np.array(valid_indices, dtype=int)

    log(f"Extracted {len(cutouts)} valid cutouts from {len(obj)} objects")

    return cutouts, fwhm_features, valid_indices


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
    cutout_radius=15,
    fwhm=None,
    threshold=0.5,
    add_score=True,
    flag_bogus=True,
    batch_size=128,
    verbose=False
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
    cutout_radius : int, optional
        Cutout radius in pixels. Default: 15 (31x31)
    fwhm : float, optional
        Image FWHM. If None, estimated from catalog.
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

    # Check if model expects FWHM input
    use_fwhm_feature = len(model.input_shape) == 2  # Two inputs: image + fwhm

    log(f"Classifying {len(obj)} objects (threshold={threshold:.2f})")

    # Extract cutouts
    cutouts, fwhm_features, valid_indices = extract_cutouts(
        obj,
        image,
        bg=bg,
        err=err,
        mask=mask,
        radius=cutout_radius,
        fwhm=fwhm,
        verbose=verbose
    )

    # Batch inference
    log(f"Running inference on {len(cutouts)} cutouts (batch_size={batch_size})")

    if use_fwhm_feature:
        predictions = model.predict(
            [cutouts, fwhm_features],
            batch_size=batch_size,
            verbose=1 if verbose else 0
        )
    else:
        predictions = model.predict(
            cutouts,
            batch_size=batch_size,
            verbose=1 if verbose else 0
        )

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
        log(f"Filtered: {len(obj_filtered)}/{len(obj)} objects retained "
            f"({100*len(obj_filtered)/len(obj):.1f}%)")

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
    verbose=True
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
    ...     training_data=(data['X'], data['y'], data['fwhm']),
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
        fwhm_features = data['fwhm']
    else:
        # Handle both tuple and dict formats
        if isinstance(training_data, dict):
            X = training_data['X']
            y = training_data['y']
            fwhm_features = training_data.get('fwhm', None)
        else:
            X, y, fwhm_features = training_data

    log(f"Training data: {len(X)} samples, {np.sum(y)} real, {len(y) - np.sum(y)} bogus")

    # Create model if not provided
    if model is None:
        input_shape = X.shape[1:]  # (height, width, channels)
        use_fwhm = fwhm_features is not None
        model = create_realbogus_model(
            input_shape=input_shape,
            use_fwhm_feature=use_fwhm
        )

    log(model.summary())

    # Calculate class weights
    if class_weight == 'balanced':
        n_real = np.sum(y)
        n_bogus = len(y) - n_real
        class_weight = {
            0: len(y) / (2 * n_bogus),
            1: len(y) / (2 * n_real)
        }
        log(f"Class weights: {class_weight}")

    # Default callbacks
    if callbacks is None:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

    # Train model
    log("Starting training...")

    if fwhm_features is not None:
        history = model.fit(
            [X, fwhm_features], y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )
    else:
        history = model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )

    log("Training complete")

    # Save model
    if model_file is not None or model_file is None:
        save_realbogus_model(model, model_file=model_file, verbose=verbose)

    return model, history


# Backwards compatibility aliases
classify_objects = classify_realbogus
create_model = create_realbogus_model
load_model = load_realbogus_model
save_model = save_realbogus_model
