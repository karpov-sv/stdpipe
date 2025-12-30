Point Spread Function (PSF) models
==================================

*STDPipe* includes basic support for point spread function (PSF) construction and analysis through the interface to `PSFEx <https://github.com/astromatic/psfex>`_ code that is able to build supersampled PSF models from the object lists created using `SExtractor <https://github.com/astromatic/sextractor>`_.
Please consider checking `its documentation <https://psfex.readthedocs.io>`__ to better understand the concepts of PSFEx operation and its possible configuration options.

The :func:`stdpipe.psf.run_psfex` routine transparently calls the `SExtractor`_ on the image and then uses its output (it needs a special set of measured object parameters and postage stamps for them) to run `PSFEx`_. The results are then parsed and returned as a structure representing the PSF model file as described `here <https://psfex.readthedocs.io/en/latest/Appendices.html#psf-file-format-description>`__. This structure may later be used to visualize PSF shape at arbitrary position inside the image, as well as to generate artificial stars and inject them into the image.

.. autofunction:: stdpipe.psf.run_psfex
   :noindex:

.. attention::

   While `PSFEx`_ may build PSF models depending on a variety of object parameters as described `here <https://psfex.readthedocs.io/en/latest/Working.html#managing-psf-variations>`__, *STDPipe* currently allows reading and interpreting only the ones depending on the position on the image (e.g. representing optical distortions). On the other hand, the original PSFEx model file may be stored to the file using `psffile` parameter of :func:`stdpipe.psf.run_psfex`, and then supplied directly to `SExtractor`_ through `psf` parameter of :func:`stdpipe.photometry.get_objects_sextractor` to be used for SExtractor PSF photometry.


Using PSF models
----------------

The PSF model built by `PSFEx`_ may also be loaded from PSFEx model file. This model may then be used to construct PSF stamps both in supersampled PSF model pixels, or original image pixels.


.. autofunction:: stdpipe.psf.load_psf
   :noindex:

.. autofunction:: stdpipe.psf.get_supersampled_psf_stamp
   :noindex:

.. autofunction:: stdpipe.psf.get_psf_stamp
   :noindex:


Placing artificial stars into the image
---------------------------------------

*STDPipe* also contains a couple of convenience functions to directly place a PSF model at a given image position - that would correspond to artificial point source injection into the image. Higher-level one, :func:`stdpipe.pipeline.place_random_stars`, inserts a number of randomly generated sources with realistically (log-uniform in fluxes) distributed brightness at random positions, as well as returns the catalogue of injected stars that may be used e.g. in deriving the object detection efficiency.

.. code-block:: python

   # Derive PSF model assuming that it does not change over the image
   psf_model = psf.run_psfex(image, mask=mask, order=0, verbose=True)

   # ..and now perform the detection efficiency analysis

   sims = [] # will hold simulated stars

   # We will repeatedly inject the stars (20 at a time), detect objects, and
   # match them in order to see whether simulated stars are detectable

   for _ in tqdm(range(100)):
       # We will operate on a copy of original image
       image1 = image.copy()

       # We will use this value as a saturation level
       saturation = 50000

       # Simulate 20 random stars
       sim = pipeline.place_random_stars(image1, psf_model, nstars=20, minflux=10, maxflux=1e6,
                wcs=wcs, gain=gain, saturation=saturation)

       # Initial metadata for injected stars
       sim['mag_calib'] = sim['mag'] + zero_fn(sim['x'], sim['y']) # Apply zero point from photometric calibration
       sim['detected'] = False
       sim['mag_measured'] = np.nan
       sim['magerr_measured'] = np.nan
       sim['flags_measured'] = np.nan

       # Mask corresponding to the saturation level
       mask1 = image1 >= saturation

       obj1 = photometry.get_objects_sextractor(image1, mask=mask|mask1, r0=1, aper=5.0,
                wcs=wcs, gain=gain, minarea=3, sn=5)
       # Apply zero point from photometric calibration
       obj1['mag_calib'] = obj1['mag'] + zero_fn(obj1['x'], obj1['y'])

       # Positional match within FWHM/2 radius
       oidx,sidx,dist = astrometry.spherical_match(obj1['ra'], obj1['dec'], sim['ra'], sim['dec'],
                pixscale*np.median(obj1['fwhm'])/2)

       # Mark matched stars
       sim['detected'][sidx] = True
       # Also store measured magnitude, its error and flags
       sim['mag_measured'][sidx] = obj1['mag_calib'][oidx]
       sim['magerr_measured'][sidx] = obj1['magerr'][oidx]
       sim['flags_measured'][sidx] = obj1['flags'][oidx]

       sims.append(sim)

   # Now we may stack the data from all runs into a single table
   from astropy.table import vstack
   sims = vstack(sims)

   # ..and plot it as a histograms of magnitudes
   h0,b0,_ = plt.hist(sims['mag_calib'], range=[12,22], bins=50, alpha=0.3, label='All simulated stars');
   h1,b1,_ = plt.hist(sims['mag_calib'][sims['detected']], range=[12,22], bins=50, alpha=0.3, label='Detected');
   h2,b2,_ = plt.hist(sims['mag_calib'][idx], range=[12,22], bins=50, alpha=0.3, label='Detected and unflagged');

   plt.legend()

   plt.xlabel('Injected magnitude')
   plt.show()

   # ..or as a detection efficiency
   plt.plot(0.5*(b0[1:]+b0[:-1]), h1/h0, 'o-', label='Detected')
   plt.plot(0.5*(b0[1:]+b0[:-1]), h2/h0, 'o-', label='Detected and unflagged')

   plt.legend()
   plt.xlabel('Injected magnitude')

   plt.title('Fraction of detected artificial stars')

The complete example of detection efficiency analysis may be seen in the `corresponding notebook <https://github.com/karpov-sv/stdpipe/blob/master/notebooks/simulated_stars.ipynb>`_.

.. attention::

   Note that the flux that you specify for the artificial star is a total flux - it will not necessarily correspond to the one measured inside an aperture (especially if the aperture size is quite small). The difference (*aperture correction*) is due to the fraction of the total flux that falls outside of your aperture, and it has to be somehow estimated and corrected if you wish to compare the generated fluxes of injected stars with measured values.

.. autofunction:: stdpipe.psf.place_psf_stamp
   :noindex:

.. autofunction:: stdpipe.pipeline.place_random_stars
   :noindex:

PSF photometry with photutils
------------------------------

*STDPipe* provides PSF fitting photometry using the photutils library as an alternative
to SExtractor. This approach is pure Python, more flexible, and works with various PSF
models including PSFEx, Gaussian, and empirical ePSF.

The :func:`stdpipe.photometry_psf.measure_objects_psf` function performs PSF photometry
at the positions of already detected objects. It supports:

- Multiple PSF model types (PSFEx, Gaussian, empirical ePSF)
- Position-dependent PSF for wide-field images
- Grouped fitting for crowded fields
- Comprehensive quality metrics

.. code-block:: python

   from stdpipe import photometry, photometry_psf, psf

   # Detect objects
   obj = photometry.get_objects_sep(image, mask=mask, thresh=3.0)

   # Build PSF model
   psf_model = psf.run_psfex(image, mask=mask, order=1, verbose=True)

   # Perform PSF photometry
   obj_psf = photometry_psf.measure_objects_psf(
       obj, image,
       psf=psf_model,
       mask=mask,
       verbose=True
   )

The output table includes fitted positions (`x_psf`, `y_psf`), fluxes (`flux`, `fluxerr`),
magnitudes (`mag`, `magerr`), and quality metrics:

- `qfit_psf` - Fit quality (0 = good, higher values indicate poor fits)
- `cfit_psf` - Central pixel fit quality
- `flags_psf` - Photutils fit flags
- `npix_psf` - Number of unmasked pixels used in fit
- `reduced_chi2_psf` - Reduced chi-squared (photutils >= 2.3.0)

.. autofunction:: stdpipe.photometry_psf.measure_objects_psf
   :noindex:


Position-dependent PSF
^^^^^^^^^^^^^^^^^^^^^^

For wide-field images where the PSF varies across the field, enable position-dependent
evaluation of PSFEx polynomial models:

.. code-block:: python

   # Build PSF model with spatial variation (order > 0)
   psf_model = psf.run_psfex(image, mask=mask, order=2, verbose=True)

   # Use position-dependent PSF
   obj_psf = photometry_psf.measure_objects_psf(
       obj, image,
       psf=psf_model,
       use_position_dependent_psf=True,  # Evaluate PSF at each position
       verbose=True
   )

This evaluates the PSFEx polynomial (constant, linear, quadratic terms) at each
source position for more accurate photometry in fields with PSF gradients.


Grouped PSF fitting for crowded fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In crowded fields with overlapping sources, grouped fitting simultaneously fits
nearby sources to reduce contamination:

.. code-block:: python

   obj_psf = photometry_psf.measure_objects_psf(
       obj, image,
       psf=psf_model,
       group_sources=True,      # Enable grouped fitting
       grouper_radius=10.0,     # Group sources within 10 pixels
       verbose=True
   )

   # Check grouping results
   print(f"Grouped {len(np.unique(obj_psf['group_id']))} groups")

This is slower but more accurate when sources are close together.


Building empirical PSF models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When PSFEx is not available or you want a purely empirical PSF, use
:func:`stdpipe.psf.create_psf_model` to build an ePSF from stars in your image:

.. code-block:: python

   from stdpipe import psf

   # Create empirical PSF from stars in the image
   epsf_model = psf.create_psf_model(
       image,
       mask=mask,
       fwhm=3.0,        # Approximate FWHM
       size=25,         # Cutout size
       oversampling=2,  # ePSF oversampling factor
       verbose=True
   )

   # Use it just like a PSFEx model
   obj_psf = photometry_psf.measure_objects_psf(
       obj, image,
       psf=epsf_model,
       verbose=True
   )

The ePSF model is built by combining postage stamps of isolated stars and returns
a PSFEx-compatible dictionary structure that works with all PSF functions.

.. autofunction:: stdpipe.psf.create_psf_model
   :noindex:

PSF photometry in SExtractor (alternative)
-------------------------------------------

As an alternative to the photutils-based approach described above, derived PSF models
may also be used for performing PSF photometry in SExtractor.

As currently *STDPipe* lacks the routine for saving PSF models back to FITS files, in order to use PSF photometry you will need to pass `psffile` argument to :func:`stdpipe.psf.run_psfex` in order to directly store the produced PSF model into the file. Then, this file may be used in SExtractor by passing its path as a `psf` argument into :func:`stdpipe.photometry.get_objects_sextractor`. It will then produce the list of detected objects having a set of additional measured parameters including `flux_psf`, `fluxerr_psf`, `mag_psf`, `magerr_psf` - they correspond to the flux derived from fitting the objects with the PSF model.

.. code-block:: python

   # We probably do not have enough stars to study PSF spatial variance, so we use order=0 here
   psf_model = psf.run_psfex(image, mask=mask, order=0, gain=gain, psffile='/tmp/psf.psf', verbose=True)

   obj_psf = photometry.get_objects_sextractor(image, mask=mask, aper=3.0, edge=10, gain=gain,
                wcs=wcs, psf='/tmp/psf.psf', verbose=False)

   # Now we may calibrate the photometry using PSF fluxes
   m_psf = pipeline.calibrate_photometry(obj_psf, cat, sr=1/3600,
                obj_col_mag='mag_psf', obj_col_mag_err='magerr_psf',
                cat_col_mag='rmag', cat_col_mag1='gmag', cat_col_mag2='rmag',
                order=0, verbose=True)
