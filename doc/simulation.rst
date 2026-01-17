Image Simulation
================

*STDPipe* provides comprehensive image simulation capabilities for generating realistic astronomical images with stars, galaxies, and various imaging artifacts. This is particularly useful for:

- Testing photometry and detection algorithms
- Training real-bogus classifiers for transient detection
- Estimating detection limits and completeness
- Validating pipeline performance

The simulation module provides both high-level and incremental APIs for maximum flexibility.

Overview
--------

The :mod:`stdpipe.simulation` module can generate:

**Astronomical Sources:**

- **Stars** with Gaussian or Moffat PSF (pixel-integrated for accuracy)
- **Galaxies** with Sersic profiles (exponential, de Vaucouleurs, etc.)

**Imaging Artifacts:**

- **Cosmic rays** (sharp, tapered, or worm-like tracks)
- **Hot pixels** (isolated or clustered)
- **Bad columns/rows** (dead, hot, or noisy)
- **Satellite trails** (with optional tumbling)
- **Diffraction spikes** (automatically added to bright stars)
- **Optical ghosts** (reflections from very bright sources)

All simulated sources are tracked in a catalog with detailed metadata, making it easy to train machine learning classifiers or validate detection algorithms.

Quick Start
-----------

The simplest way to create a simulated image is using the high-level :func:`~stdpipe.simulation.simulate_image` function:

.. code-block:: python

   from stdpipe import simulation

   # Create a realistic simulated image
   result = simulation.simulate_image(
       width=500,
       height=500,
       n_stars=100,
       n_galaxies=20,
       n_cosmic_rays=10,
       background=1000.0,
       readnoise=10.0
   )

   image = result['image']      # The simulated image
   catalog = result['catalog']  # Catalog of all injected sources

   print(f"Created {len(catalog)} objects")
   print(f"Image shape: {image.shape}")

The catalog contains all injected sources with their properties and a ``type`` column to distinguish between different object types.

High-Level API
--------------

Single-Call Image Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~stdpipe.simulation.simulate_image` function creates a complete simulated image in a single call:

.. code-block:: python

   from stdpipe import simulation
   import numpy as np

   # Create a complex simulated field
   result = simulation.simulate_image(
       width=1000,
       height=1000,
       # Stars
       n_stars=200,
       star_flux_range=(100, 50000),
       star_fwhm=3.5,
       star_psf='moffat',           # or 'gaussian'
       star_beta=2.5,               # Moffat beta parameter
       # Galaxies
       n_galaxies=50,
       galaxy_flux_range=(500, 10000),
       galaxy_r_eff_range=(3, 15),
       galaxy_n_range=(0.5, 4.0),   # Sersic index
       galaxy_ellipticity_range=(0.0, 0.7),
       # Artifacts
       n_cosmic_rays=15,
       cosmic_ray_length_range=(10, 60),
       n_hot_pixels=30,
       n_satellites=2,
       satellite_length_range=(200, 500),
       n_bad_columns=1,
       # Advanced features
       diffraction_spikes=True,     # Add spikes to bright stars
       spike_threshold=50000,       # Flux threshold for spikes
       optical_ghosts=True,         # Add ghosts to very bright stars
       ghost_threshold=100000,
       # Noise model
       background=1000.0,
       readnoise=10.0,
       gain=1.0,
       # Geometry
       edge=20,                     # Keep sources away from edges
       wcs=None,                    # Optional WCS for RA/Dec
       verbose=True
   )

   image = result['image']
   catalog = result['catalog']
   background = result['background']
   noise = result['noise']

   # Analyze the catalog
   stars = catalog[catalog['type'] == 'star']
   galaxies = catalog[catalog['type'] == 'galaxy']
   artifacts = catalog[catalog['is_real'] == False]

   print(f"Stars: {len(stars)}")
   print(f"Galaxies: {len(galaxies)}")
   print(f"Artifacts: {len(artifacts)}")

The ``is_real`` column in the catalog distinguishes astronomical sources (``True``) from artifacts (``False``), which is perfect for training real-bogus classifiers.

Incremental API
---------------

For more control, you can build images incrementally by adding components one at a time.

Adding Stars
^^^^^^^^^^^^

Use :func:`~stdpipe.simulation.add_stars` to add stars with customizable PSF:

.. code-block:: python

   from stdpipe import simulation
   import numpy as np

   # Create blank image
   image = np.zeros((500, 500))

   # Add stars with Gaussian PSF
   catalog_gauss = simulation.add_stars(
       image,
       n=50,
       flux_range=(1000, 20000),
       fwhm=3.0,
       psf_type='gaussian',
       edge=20
   )

   # Add more stars with Moffat PSF
   catalog_moffat = simulation.add_stars(
       image,
       n=30,
       flux_range=(500, 10000),
       fwhm=4.0,
       psf_type='moffat',
       beta=2.5,
       diffraction_spikes=True,    # Add spikes to bright stars
       spike_threshold=15000,
       optical_ghosts=True,        # Add ghosts to very bright stars
       ghost_threshold=50000
   )

   print(f"Added {len(catalog_gauss)} Gaussian stars")
   print(f"Added {len(catalog_moffat)} Moffat stars")

.. note::

   The pixel-integrated PSF models eliminate systematic flux biases that can affect photometry testing. For Gaussian PSF, the bias is reduced from +11.6% at FWHM=1.5 to <0.1% for all FWHM values.

Adding Galaxies
^^^^^^^^^^^^^^^

Use :func:`~stdpipe.simulation.add_galaxies` to add extended sources with Sersic profiles:

.. code-block:: python

   # Add galaxies with various morphologies
   catalog = simulation.add_galaxies(
       image,
       n=20,
       flux_range=(1000, 10000),
       r_eff_range=(5, 20),          # Effective radius in pixels
       n_range=(0.5, 4.0),           # Sersic index
       ellipticity_range=(0.0, 0.8),
       edge=50,                      # Keep away from edges
       gain=1.0
   )

   # Examine galaxy properties
   for galaxy in catalog:
       print(f"Galaxy at ({galaxy['x']:.1f}, {galaxy['y']:.1f}): "
             f"n={galaxy['sersic_n']:.1f}, r_eff={galaxy['r_eff']:.1f}, "
             f"e={galaxy['ellipticity']:.2f}, PA={galaxy['position_angle']:.1f}°")

The Sersic index ``n`` controls the profile shape:

- ``n=0.5``: Gaussian-like (very compact)
- ``n=1.0``: Exponential disk (spiral galaxies)
- ``n=4.0``: de Vaucouleurs (elliptical galaxies)

Adding Artifacts
^^^^^^^^^^^^^^^^

Add various imaging artifacts to test detection and classification algorithms:

.. code-block:: python

   # Add cosmic ray tracks
   cat_cosmic = simulation.add_cosmic_rays(
       image,
       n_rays=10,
       length_range=(10, 50),
       width_range=(1, 3),
       intensity_range=(5000, 20000),
       profile='sharp'              # or 'tapered', 'worm'
   )

   # Add hot pixels
   cat_hot = simulation.add_hot_pixels(
       image,
       n_pixels=20,
       intensity_range=(1000, 5000),
       clustering=True,             # Create clusters
       cluster_size=3
   )

   # Add satellite trails
   cat_satellites = simulation.add_satellite_trails(
       image,
       n_trails=2,
       length_range=(200, 400),
       width_range=(3, 8),
       intensity_range=(10000, 30000),
       profile='gaussian',          # or 'linear'
       tumbling_prob=0.3            # Fraction with intensity variations
   )

   # Add bad columns
   cat_bad = simulation.add_bad_columns(
       image,
       n_columns=1,
       bad_type='dead',             # or 'hot', 'noisy'
       orientation='vertical'       # or 'horizontal'
   )

PSF Models
----------

Gaussian PSF
^^^^^^^^^^^^

The default PSF model is a pixel-integrated Gaussian:

.. code-block:: python

   from stdpipe.simulation import create_psf_stamp

   # Create a Gaussian PSF stamp
   psf = create_psf_stamp(
       size=25,                     # Stamp size (odd number)
       x0=12.3,                     # Center X (can be sub-pixel)
       y0=12.7,                     # Center Y
       fwhm=3.5,
       psf_type='gaussian',
       pixel_integrated=True        # Accurate flux conservation
   )

   print(f"PSF sum: {psf.sum():.6f}")  # Should be 1.0

Moffat PSF
^^^^^^^^^^

The Moffat PSF has broader wings than Gaussian, which is more realistic for many telescopes:

.. code-block:: python

   # Create a Moffat PSF stamp
   psf = create_psf_stamp(
       size=31,
       x0=15.0,
       y0=15.0,
       fwhm=4.0,
       psf_type='moffat',
       beta=2.5,                    # Beta parameter (DAOPHOT 'moffat25')
       pixel_integrated=True
   )

The ``beta`` parameter controls the wing profile:

- ``beta=1.5``: Very extended wings
- ``beta=2.5``: Standard (DAOPHOT 'moffat25')
- ``beta=4.0``: Compact, closer to Gaussian

Galaxy Profiles
---------------

Creating Custom Galaxy Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create galaxy models with custom Sersic profiles:

.. code-block:: python

   from stdpipe.simulation import create_sersic_profile, place_galaxy
   import numpy as np

   # Create a Sersic profile stamp
   profile = create_sersic_profile(
       size=101,
       x0=50.0,
       y0=50.0,
       amplitude=1.0,
       r_eff=10.0,                 # Effective radius (half-light)
       n=1.0,                      # Sersic index
       ellipticity=0.5,            # 0=circular, 1=line
       position_angle=45.0         # Degrees, 0=vertical
   )

   # Place a galaxy into an image
   image = np.zeros((200, 200))
   place_galaxy(
       image,
       x0=100.5,
       y0=100.5,
       flux=10000.0,               # Total integrated flux
       r_eff=15.0,
       n=4.0,                      # de Vaucouleurs profile
       ellipticity=0.6,
       position_angle=120.0,
       gain=1.0
   )

Common Sersic Profiles
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Exponential disk (spiral galaxy)
   cat_spirals = simulation.add_galaxies(
       image, n=10, n_range=(0.8, 1.2), r_eff_range=(10, 20)
   )

   # Elliptical galaxies (de Vaucouleurs)
   cat_ellipticals = simulation.add_galaxies(
       image, n=5, n_range=(3.5, 4.5), r_eff_range=(5, 15),
       ellipticity_range=(0.3, 0.8)
   )

   # Compact sources (dwarf galaxies)
   cat_dwarfs = simulation.add_galaxies(
       image, n=15, n_range=(0.5, 1.0), r_eff_range=(2, 5),
       flux_range=(200, 1000)
   )

Real-Bogus Classifier Training
-------------------------------

The simulation module is ideal for training real-bogus classifiers because every object is labeled:

.. code-block:: python

   from stdpipe import simulation, photometry
   import numpy as np

   # Generate training set
   result = simulation.simulate_image(
       width=2000,
       height=2000,
       n_stars=500,
       star_flux_range=(100, 100000),
       n_galaxies=100,
       n_cosmic_rays=50,
       n_hot_pixels=100,
       n_satellites=5,
       background=1000.0,
       readnoise=10.0,
       verbose=True
   )

   image = result['image']
   true_catalog = result['catalog']

   # Detect objects using STDPipe
   detected = photometry.get_objects_sep(image, thresh=5.0)

   # Match detected objects to truth
   from scipy.spatial import cKDTree

   # Build KD-tree for truth catalog
   truth_coords = np.c_[true_catalog['x'], true_catalog['y']]
   tree = cKDTree(truth_coords)

   # Match detected objects
   detected_coords = np.c_[detected['x'], detected['y']]
   distances, indices = tree.query(detected_coords)

   # Label detected objects
   detected['is_real'] = False
   detected['matched_type'] = 'none'

   match_radius = 3.0  # pixels
   matched = distances < match_radius

   detected['is_real'][matched] = true_catalog['is_real'][indices[matched]]
   detected['matched_type'][matched] = true_catalog['type'][indices[matched]]

   # Now you have labeled training data for ML classifiers
   real_sources = detected[detected['is_real'] == True]
   artifacts = detected[detected['is_real'] == False]

   print(f"Real sources: {len(real_sources)}")
   print(f"Artifacts: {len(artifacts)}")
   print(f"Unmatched detections: {np.sum(~matched)}")

   # Extract features for classifier training
   features = np.c_[
       detected['fwhm'],
       detected['a'],              # Semi-major axis
       detected['b'],              # Semi-minor axis
       detected['flux'],
       detected['flags']
   ]

   labels = detected['is_real'].astype(int)

   # Train your classifier here...
   # from sklearn.ensemble import RandomForestClassifier
   # clf = RandomForestClassifier()
   # clf.fit(features, labels)

Testing Detection Limits
-------------------------

Simulate images at various flux levels to measure detection completeness:

.. code-block:: python

   from stdpipe import simulation, photometry
   import numpy as np
   import matplotlib.pyplot as plt

   # Flux levels to test
   flux_levels = np.logspace(2, 4, 20)  # 100 to 10000 ADU
   completeness = []

   for flux in flux_levels:
       # Simulate field with stars at this flux level
       result = simulation.simulate_image(
           width=1000,
           height=1000,
           n_stars=100,
           star_flux_range=(flux * 0.9, flux * 1.1),
           star_fwhm=3.0,
           background=1000.0,
           readnoise=10.0
       )

       image = result['image']
       true_cat = result['catalog']
       true_stars = true_cat[true_cat['type'] == 'star']

       # Detect objects
       detected = photometry.get_objects_sep(image, thresh=5.0)

       # Match and compute completeness
       from scipy.spatial import cKDTree
       tree = cKDTree(np.c_[true_stars['x'], true_stars['y']])
       distances, _ = tree.query(np.c_[detected['x'], detected['y']])
       n_recovered = np.sum(distances < 3.0)

       completeness.append(n_recovered / len(true_stars))
       print(f"Flux {flux:.0f}: {completeness[-1]*100:.1f}% recovered")

   # Plot completeness curve
   plt.figure(figsize=(8, 6))
   plt.semilogx(flux_levels, completeness, 'o-')
   plt.xlabel('Flux (ADU)')
   plt.ylabel('Completeness')
   plt.title('Detection Completeness vs Flux')
   plt.grid(True, alpha=0.3)
   plt.ylim(0, 1.1)
   plt.show()

Testing Photometry Accuracy
----------------------------

Validate photometry algorithms using simulated sources with known flux:

.. code-block:: python

   from stdpipe import simulation, photometry, photometry_measure
   import numpy as np

   # Simulate isolated stars with known fluxes
   result = simulation.simulate_image(
       width=1000,
       height=1000,
       n_stars=50,
       star_flux_range=(5000, 20000),
       star_fwhm=3.5,
       star_psf='gaussian',
       n_galaxies=0,
       n_cosmic_rays=0,
       background=1000.0,
       readnoise=10.0,
       edge=50                     # Keep stars isolated
   )

   image = result['image']
   true_catalog = result['catalog']
   true_stars = true_catalog[true_catalog['type'] == 'star']

   # Detect and measure
   detected = photometry.get_objects_sep(image, thresh=5.0)
   measured = photometry_measure.measure_objects(
       detected, image, fwhm=3.5, aper=10.0
   )

   # Match measured to truth
   from scipy.spatial import cKDTree
   tree = cKDTree(np.c_[true_stars['x'], true_stars['y']])
   distances, indices = tree.query(np.c_[measured['x'], measured['y']])

   # Analyze photometry accuracy
   matched = distances < 2.0
   measured_matched = measured[matched]
   true_matched = true_stars[indices[matched]]

   # Compare fluxes
   flux_ratio = measured_matched['flux'] / true_matched['flux']
   flux_bias = np.median(flux_ratio) - 1.0
   flux_scatter = np.std(flux_ratio)

   print(f"Matched {np.sum(matched)}/{len(true_stars)} stars")
   print(f"Photometry bias: {flux_bias*100:.2f}%")
   print(f"Photometry scatter: {flux_scatter*100:.2f}%")

   # Plot results
   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 5))

   plt.subplot(1, 2, 1)
   plt.plot(true_matched['flux'], measured_matched['flux'], 'o', alpha=0.5)
   plt.plot([0, 25000], [0, 25000], 'r--', label='Perfect')
   plt.xlabel('True Flux (ADU)')
   plt.ylabel('Measured Flux (ADU)')
   plt.title('Photometry Accuracy')
   plt.legend()
   plt.grid(True, alpha=0.3)

   plt.subplot(1, 2, 2)
   plt.hist(flux_ratio, bins=20, alpha=0.7)
   plt.axvline(1.0, color='r', linestyle='--', label='Perfect')
   plt.axvline(np.median(flux_ratio), color='g', linestyle='-', label='Median')
   plt.xlabel('Measured / True Flux')
   plt.ylabel('Count')
   plt.title(f'Bias: {flux_bias*100:.2f}%')
   plt.legend()
   plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Catalog Format
--------------

The catalog returned by simulation functions is an Astropy Table with the following columns:

Common Columns (all objects)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``x``, ``y`` - Image coordinates
- ``type`` - Object type: 'star', 'galaxy', 'cosmic_ray', 'hot_pixel', 'satellite_trail', 'bad_column', 'bad_row'
- ``is_real`` - Boolean: ``True`` for astronomical sources, ``False`` for artifacts
- ``ra``, ``dec`` - Sky coordinates (if WCS provided, else NaN)

Star-Specific Columns
^^^^^^^^^^^^^^^^^^^^^

- ``flux`` - Total flux in ADU
- ``mag`` - Instrumental magnitude (-2.5 * log10(flux))
- ``fwhm`` - PSF FWHM in pixels
- ``psf_type`` - PSF model used ('gaussian' or 'moffat')

Galaxy-Specific Columns
^^^^^^^^^^^^^^^^^^^^^^^

- ``flux`` - Total integrated flux
- ``r_eff`` - Effective (half-light) radius in pixels
- ``sersic_n`` - Sersic index
- ``ellipticity`` - Ellipticity (0=circular, 1=line)
- ``position_angle`` - Position angle in degrees

Artifact-Specific Columns
^^^^^^^^^^^^^^^^^^^^^^^^^^

For cosmic rays and satellites:

- ``length`` - Track length in pixels
- ``width`` - Track width in pixels
- ``angle`` - Track angle in degrees
- ``intensity`` - Peak intensity

For hot pixels:

- ``intensity`` - Pixel intensity

For bad columns:

- ``position`` - Column/row index
- ``bad_type`` - Type: 'dead', 'hot', or 'noisy'
- ``orientation`` - 'vertical' or 'horizontal'

API Reference
-------------

High-Level Functions
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: stdpipe.simulation.simulate_image
   :noindex:

Incremental Functions
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: stdpipe.simulation.add_stars
   :noindex:

.. autofunction:: stdpipe.simulation.add_galaxies
   :noindex:

.. autofunction:: stdpipe.simulation.add_cosmic_rays
   :noindex:

.. autofunction:: stdpipe.simulation.add_hot_pixels
   :noindex:

.. autofunction:: stdpipe.simulation.add_satellite_trails
   :noindex:

.. autofunction:: stdpipe.simulation.add_bad_columns
   :noindex:

PSF Functions
^^^^^^^^^^^^^

.. autofunction:: stdpipe.simulation.create_psf_stamp
   :noindex:

.. autofunction:: stdpipe.simulation.create_moffat_psf
   :noindex:

Galaxy Functions
^^^^^^^^^^^^^^^^

.. autofunction:: stdpipe.simulation.create_sersic_profile
   :noindex:

.. autofunction:: stdpipe.simulation.place_galaxy
   :noindex:

Artifact Functions
^^^^^^^^^^^^^^^^^^

.. autofunction:: stdpipe.simulation.create_cosmic_ray
   :noindex:

.. autofunction:: stdpipe.simulation.create_satellite_trail
   :noindex:

.. autofunction:: stdpipe.simulation.create_diffraction_spikes
   :noindex:

.. autofunction:: stdpipe.simulation.create_optical_ghost
   :noindex:
