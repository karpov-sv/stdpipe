Object detection and measurement
================================

This page covers object detection and flux measurement (aperture, optimal extraction, SEP-based, and PSF photometry). For photometric *calibration* (zero points, color terms) see :doc:`photometry`, for PSF model construction see :doc:`psf`, and for detection flag definitions see :doc:`flags`.

Detecting objects on the image
------------------------------

*STDPipe* provides three functions for detecting objects on the image:

- :func:`stdpipe.photometry.get_objects_sextractor` - based on external `SExtractor <https://github.com/astromatic/sextractor>`_ binary
- :func:`stdpipe.photometry.get_objects_sep` - based on Python `SEP <https://github.com/kbarbary/sep>`_ library
- :func:`stdpipe.photometry.get_objects_photutils` - based on `photutils <https://photutils.readthedocs.io/>`_ library (pure Python, no compiled dependencies)

The first two have mostly identical signatures and arguments, but differ in minor details like meaning of returned detection flags etc.

The detection in both cases is based on building the noise model through (grid-based) background and background rms estimation, and then extracting the groups of connected pixels above some pre-defined threshold. Optionally, before the thresholding, the image may be smoothed with some small kernel in order to improve the detection of fainter objects. We recommend checking the `SExtractor documentation <https://sextractor.readthedocs.io/>`_ to better understand the concepts of it.

Also, the routines may automatically reject the objects detected too close to frame edge, and probably truncated or improperly measured - this is controlled by `edge` argument. Less significant detections - with :math:`\frac1{\rm magerr} < {\rm S/N}` - may also be automatically rejected by providing the `sn` argument.

Both routines return the results as a standard Astropy Table, ordered by the object brightness. The table contains at least the following columns:

-  `x`, `y`, `xerr`, `yerr` - positions of the objects in image coordinates
-  `ra`, `dec` - positions of the objects in sky coordinates according to provided WCS
-  `a`, `b`, `theta` - object ellipse semimajor and semiminor axes, as well as its position angle in image coordinates
-  `flux`, `fluxerr` - measured flux and its error in a circular aperture
-  `mag`, `magerr` - instrumental magnitude computed as `-2.5*log10(flux)`, and its error computed as `2.5/log(10)*fluxerr/flux`
-  `flags` - detection flags
-  `fwhm` - per-object FWHM estimation

For :func:`~stdpipe.photometry.get_objects_sextractor`, the output may also contain columns related to PSF photometry (`x_psf`, `y_psf`, `flux_psf`, `fluxerr_psf`, `mag_psf`, `magerr_psf`, `chi2_psf`, `spread_model`, `spreaderr_model`), as well as any additional measuremet parameter requested through `extra_params` argument.

.. attention::

   *STDPipe* (as well as SEP library) uses pixel coordinate convention with `(0, 0)` as the origin - that differs from *SExtractor* that uses `(1, 1)` as origin of coordinates! So the routine transparently converts `x` and `y` (as well as `x_psf` and `y_psf`) in the output from *SExtractor* to proper origin, so that the coordinates of objects detected by both routines are in the same system. However, if you manually add some pixel coordinate parameter to the output through `extra_params` argument, they will not be appropriately adjusted!

Below are some examples of object detection.

.. code-block:: python

   # We will detect objects using SExtractor and get their measurements
   # in apertures with 3 pixels radius.
   # We will also ignore anything closer than 10 pixels to image edge
   obj = photometry.get_objects_sextractor(image, mask=mask, aper=3.0, gain=gain, edge=10)
   print(len(obj), 'objects detected')

   # Rough estimation of average FWHM of detected objects, taking into account
   # only unflagged (e.g. not saturated) ones
   fwhm = np.median(obj['fwhm'][obj['flags'] == 0])
   print('Average FWHM is %.1f pixels' % fwhm)

Next example shows how to receive extra columns, as well as checkimages, from SExtractor

.. code-block:: python

   # Now we will also get segmentation map and a column with object numbers correspoding to this map
   obj,segm = photometry.get_objects_sextractor(image, mask=mask,
                aper=3.0, gain=gain, edge=10, extra_params=['NUMBER'],
                checkimages=['SEGMENTATION'], verbose=True)

   # We may now e.g. completely mask the footprints of objects having masked pixels
   fmask = np.zeros_like(mask)
   for _ in obj[(obj['flags'] & 0x100) > 0]:
       fmask |= segm == _['NUMBER']

Finally, using SExtractor star/galaxy separators - `CLASS_STAR` and `SPREAD_MODEL`. The latter require PSF model - so we will build it using :func:`stdpipe.psf.run_psfex` documented elsewhere

.. code-block:: python

   # Get PSF model and store it to temporary file
   psf_model = psf.run_psfex(image, mask=mask, order=0, gain=gain, psffile='/tmp/psf.psf', verbose=True)

   # Run SExtractor
   # You should provide correct path to default.nnw file from SExtractor installation on your system!
   obj = photometry.get_objects_sextractor(image, mask=mask, edge=10, wcs=wcs,
                aper=3.0, extra_params=['CLASS_STAR', 'NUMBER'],
                extra={'SEEING_FWHM':fwhm, 'STARNNW_NAME':'/Users/karpov/opt/miniconda3/envs/stdpipe//share/sextractor/default.nnw'},
                psf='/tmp/psf.psf', verbose=True)

   for i,cand in enumerate(obj):
       print('Candidate %d at x/y = %.1f %.1d and RA/Dec = %.4f %.4f' %
                (i, cand['x'], cand['y'], cand['ra'], cand['dec']))

       print('SPREAD_MODEL = %.3f +/- %.3f, CLASS_STAR = %.2f' %
                (cand['spread_model'], cand['spreaderr_model'], cand['CLASS_STAR']))

.. note::

   The most important problem with object detection using these routines is handling of blended objects, as the codes we are using can't properly deblend close groups, except for simplest cases.

.. autofunction:: stdpipe.photometry.get_objects_sextractor
   :noindex:

.. autofunction:: stdpipe.photometry.get_objects_sep
   :noindex:


Detection using photutils
^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~stdpipe.photometry.get_objects_photutils` function provides a pure-Python alternative using the `photutils <https://photutils.readthedocs.io/>`_ library. It offers three detection algorithms:

- **segmentation** (default) - Image segmentation with connected pixels above threshold. Supports optional multi-level deblending for crowded fields.
- **dao** - DAOStarFinder algorithm using Gaussian convolution, optimized for stellar fields with known FWHM.
- **iraf** - IRAFStarFinder, similar to DAOStarFinder but with IRAF-compatible behavior.

The segmentation method with deblending is particularly useful for crowded fields where sources overlap:

.. code-block:: python

   # Detect objects using photutils segmentation with deblending
   obj = photometry.get_objects_photutils(image, mask=mask, thresh=3.0,
                method='segmentation', deblend=True, aper=3.0, edge=10)
   print(len(obj), 'objects detected')

   # For stellar fields, DAOStarFinder may be faster
   obj = photometry.get_objects_photutils(image, mask=mask, thresh=5.0,
                method='dao', fwhm=3.0, aper=3.0)

.. autofunction:: stdpipe.photometry.get_objects_photutils
   :noindex:


FWHM estimation
----------------

After detection, you typically need a robust estimate of the image FWHM for aperture sizing, optimal extraction, PSF photometry, and matching radius selection. *STDPipe* provides two functions for this:

- :func:`~stdpipe.photometry.estimate_fwhm_from_objects` — higher-level, works directly on detection tables with automatic quality filtering (S/N, flags, ellipticity)
- :func:`~stdpipe.photometry.estimate_fwhm` — lower-level, operates on a raw array of FWHM values

Both use a sliding-window mode estimator that is resistant to outliers from galaxies and blends. When ``flux_radius`` is available (from :func:`~stdpipe.photometry.get_objects_sep`), it is preferred as it is 2-3x more stable than Gaussian-core FWHM.

.. code-block:: python

   # Higher-level: automatic quality filtering
   fwhm = photometry.estimate_fwhm_from_objects(obj, verbose=True)
   print('FWHM: %.2f pixels' % fwhm)

   # Lower-level: manual filtering + raw array
   good = (obj['flags'] == 0) & (obj['magerr'] < 0.1)
   fwhm = photometry.estimate_fwhm(obj['fwhm'], good=good)


Position-dependent FWHM
^^^^^^^^^^^^^^^^^^^^^^^

Wide-field images often show FWHM variation across the field from focus gradients, optical aberrations, or tracking. Passing ``spatial_order >= 1`` fits a 2-D polynomial ``FWHM(x, y)`` to the per-object values (with ``a*b`` weights and sigma-clipping) and returns a :class:`~stdpipe.photometry.FWHMMap` callable instead of a scalar. The callable evaluates the model at any position, and ``float(fwhm_map)`` gives a scalar median summary for APIs that require one.

.. code-block:: python

   # Fit FWHM(x, y) as a 2-D polynomial
   fwhm_map = photometry.estimate_fwhm_from_objects(
       obj, spatial_order=2, image_shape=image.shape, verbose=True,
   )
   print('median FWHM:', float(fwhm_map), 'pix')
   print('FWHM at image centre:', fwhm_map(image.shape[1] / 2, image.shape[0] / 2))

If too few high-quality candidates survive for the requested order, the function transparently falls back to the scalar path.

:func:`~stdpipe.photometry.get_objects_sep` exposes the same feature via the ``fwhm_spatial_order`` parameter. When set, aperture and background-annulus radii are scaled per source and the per-source width is passed to ``sep.sum_circle_optimal`` (the windowed centroider still uses the scalar summary). The fitted model is stored in ``obj.meta['fwhm_phot_model']``:

.. code-block:: python

   obj = photometry.get_objects_sep(image, fwhm=True, fwhm_spatial_order=2,
                                    optimal=True, aper=1.5, bkgann=(3, 5))
   fmap = obj.meta['fwhm_phot_model']  # FWHMMap, or absent for scalar order=0

The downstream measurement routines :func:`~stdpipe.photometry.measure_objects` and :func:`~stdpipe.photometry_measure.measure_objects_sep` also accept a callable ``fwhm``. The SEP backend broadcasts a per-source width and per-source aperture/bkgann arrays directly to SEP; the pure-Python backend uses the per-source width in ungrouped optimal extraction and falls back to the scalar median where the underlying photutils APIs require a single radius.

.. autofunction:: stdpipe.photometry.estimate_fwhm_from_objects
   :noindex:

.. autoclass:: stdpipe.photometry.FWHMMap
   :members: __call__, __float__
   :noindex:

.. autofunction:: stdpipe.photometry.estimate_fwhm
   :noindex:


More accurate photometric measurements
--------------------------------------

The photometric measurements returned by detection routines are sometimes not optimal. The :func:`stdpipe.photometry.measure_objects` function provides several improvements:

- **Local background estimation** using annuli around each object instead of global background model
- **Centroiding** to refine object positions before measurement
- **Optimal extraction** (Naylor 1998) for ~10% S/N improvement on point sources
- **Grouped fitting** for crowded fields with overlapping PSFs


Local background estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The detection routines use a globally estimated background model (built as a low-resolution spatial map and then interpolated to original pixels, see `here <https://sextractor.readthedocs.io/en/latest/Background.html>`_). On rapidly varying backgrounds, you may expect better results from locally estimated background using annuli around every object:

.. code-block:: python

   # We will pass FWHM so that aperture and background radii are relative to it
   # We will also reject all objects with measured S/N < 5
   obj = photometry.measure_objects(obj, image, mask=mask, fwhm=fwhm, gain=gain,
                aper=1.0, bkgann=[5, 7], sn=5, verbose=True)
   print(len(obj), 'objects properly measured')


Centroiding
^^^^^^^^^^^

You can refine object positions before photometry using iterative centroiding. This improves aperture placement, especially when initial positions from detection are approximate:

.. code-block:: python

   # Refine positions with 3 centroiding iterations before aperture photometry
   obj = photometry.measure_objects(obj, image, mask=mask, fwhm=fwhm,
                aper=1.0, centroid_iter=3, verbose=True)

   # Original positions are stored in x_orig, y_orig columns
   print('Position shift:', np.median(np.hypot(obj['x'] - obj['x_orig'],
                                                obj['y'] - obj['y_orig'])))


Optimal extraction
^^^^^^^^^^^^^^^^^^

For point sources, optimal extraction (`Naylor 1998 <https://ui.adsabs.harvard.edu/abs/1998MNRAS.296..339N>`_) provides ~10% improvement in signal-to-noise ratio by using PSF-weighted photometry instead of simple aperture sums:

.. code-block:: python

   # Use optimal extraction with Gaussian PSF based on FWHM
   obj = photometry.measure_objects(obj, image, mask=mask,
                fwhm=fwhm, aper=1.5, optimal=True, verbose=True)

   # Or provide a PSF model from PSFEx for better accuracy
   psf_model = psf.run_psfex(image, mask=mask, gain=gain)
   obj = photometry.measure_objects(obj, image, mask=mask,
                psf=psf_model, aper=1.5, optimal=True, verbose=True)

Optimal extraction adds quality metrics to the output:

- ``chi2_optimal`` - chi-squared of the fit (lower is better)
- ``norm_optimal`` - PSF normalization factor (should be ~1 for point sources)
- ``npix_optimal`` - number of unmasked pixels used in the fit


Grouped optimal extraction for crowded fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In crowded fields where source PSFs overlap, grouped optimal extraction simultaneously fits fluxes for nearby sources using weighted least squares. This properly accounts for flux sharing between overlapping PSFs:

.. code-block:: python

   # Grouped extraction - sources within 2*FWHM are fitted together
   obj = photometry.measure_objects(obj, image, mask=mask,
                fwhm=fwhm, aper=1.5, optimal=True,
                group_sources=True, grouper_radius=2*fwhm,
                verbose=True)

   # Check group information
   print('Sources in groups of 2+:', len(obj[obj['group_size'] > 1]))

This adds ``group_id`` and ``group_size`` columns to identify which sources were fitted together.


.. autofunction:: stdpipe.photometry.measure_objects
   :noindex:


SEP-based photometry
^^^^^^^^^^^^^^^^^^^^

The :func:`stdpipe.photometry_measure.measure_objects_sep` function provides an alternative measurement backend using SEP 1.4+ features. It offers sigma-clipped local background estimation and grouped optimal extraction implemented in C for better performance.

.. code-block:: python

   from stdpipe.photometry_measure import measure_objects_sep, _HAS_SEP_OPTIMAL

   if _HAS_SEP_OPTIMAL:
       # Aperture photometry with robust local background
       obj = measure_objects_sep(obj, image, aper=1.5, fwhm=fwhm,
                    bkgann=(3.0, 5.0), sn=5, verbose=True)

       # Grouped optimal extraction for crowded fields
       obj = measure_objects_sep(obj, image, aper=1.5, fwhm=fwhm,
                    optimal=True, group_sources=True, sn=5, verbose=True)

.. autofunction:: stdpipe.photometry_measure.measure_objects_sep
   :noindex:


PSF photometry
^^^^^^^^^^^^^^

For the most accurate flux measurements, especially in crowded fields, PSF fitting photometry is available through two backends:

- :func:`stdpipe.photometry_psf.measure_objects_psf` — photutils backend (pure Python, flexible PSF models)
- :func:`stdpipe.photometry_iraf.measure_objects_psf` — DAOPHOT backend (classic IRAF, requires PyRAF)

Both support grouped fitting for crowded fields and return fitted positions alongside fluxes. See :doc:`psf` for detailed documentation on PSF model construction, position-dependent PSF, grouped fitting, and empirical ePSF models.

.. code-block:: python

   from stdpipe import photometry_psf

   # PSF photometry with automatic Gaussian PSF (photutils backend)
   result = photometry_psf.measure_objects_psf(obj, image, fwhm=fwhm)

   # With PSFEx model and grouped fitting
   result = photometry_psf.measure_objects_psf(obj, image, psf=psf_model,
                group_sources=True, grouper_radius=10.0)

   # DAOPHOT backend (requires PyRAF/IRAF)
   from stdpipe import photometry_iraf
   result = photometry_iraf.measure_objects_psf(obj, image, fwhm=fwhm)
