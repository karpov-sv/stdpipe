Object detection and measurement
================================

Detecting objects on the image
------------------------------

*STDPipe* currently contains two functions for detecting objects on the image, one based on external `SExtractor <https://github.com/astromatic/sextractor>`_ binary (:func:`stdpipe.photometry.get_objects_sextractor`), other - on Python `SEP <https://github.com/kbarbary/sep>`_ library (:func:`stdpipe.photometry.get_objects_sextractor`). They have mostly identical signatures and arguments, but differ in minor details like meaning of returned detection flags etc.

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

More accurate photometric measurements
--------------------------------------

The photometric measurements returned by the routines above are sometimes not the best ones you may extract from the image. E.g. they are based on globally estimated background model (built as a low-resolution spatial map and then intepolated to original pixels, see `here <https://sextractor.readthedocs.io/en/latest/Background.html>`_). On a rapidly varying backgrounds, you may expect better results from locally estimated background - e.g using local annuli around every object and sigma-clipped averages of pixel values inside them. We have a function :func:`stdpipe.photometry.measure_objects` that tries to compute it.

.. code-block:: python

   # We will pass this FWHM to measurement function so that aperture and
   # background radii will be relative to it.
   # We will also reject all objects with measured S/N < 5
   obj = photometry.measure_objects(obj, image, mask=mask, fwhm=fwhm, gain=gain,
                aper=1.0, bkgann=[5, 7], sn=5, verbose=True)
   print(len(obj), 'objects properly measured')


.. autofunction:: stdpipe.photometry.measure_objects
   :noindex:
