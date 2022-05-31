Transient detection and filtering
=================================

Transients may be detected either on initial *science* image as described in :ref:`Object detection and measurement`, or on the difference image after image subtraction. For the latter, you may also make use of a difference image noise model, as returned by :func:`stdpipe.subtraction.run_hotpants` with :code:`get_noise=True` - it will reduce the amount of subtraction artefacts at the position of brighter objects.

.. code-block:: python

   # Get PSF model of and store it to temporary file
   psf_model = psf.run_psfex(image, mask=mask, order=0, gain=gain, psffile='/tmp/psf.psf', verbose=True)

   # Run SExtractor on difference image with custom noise model,
   # using both image and template masks for masking
   sobj = photometry.get_objects_sextractor(diff, mask=mask|tmask, err=ediff,
                edge=10, wcs=wcs, aper=5.0, psf='/tmp/psf.psf', verbose=True)

   # Perform forced aperture photometry in a circular aperture with 1.0*FWHM radius,
   # measuring local backround in an annulus between 5.0*FWHM and 7.0*FWHM,
   # and again with custom noise model and forced to zero global background level
   sobj = photometry.measure_objects(sobj, diff, mask=mask|tmask, fwhm=fwhm, aper=1.0, bkgann=[5, 7],
                sn=3, verbose=True, bg=0, err=ediff)

   # The difference is in original image normalization, so we know photometric zero point
   sobj['mag_calib'] = sobj['mag'] + m['zero_fn'](sobj['x'], sobj['y'])
   sobj['mag_calib_err'] = np.hypot(sobj['magerr'], m['zero_fn'](sobj['x'], sobj['y'], get_err=True))

   # We may immediately reject flagged objects as they correspond to imaging artefacts (masked regions)
   sobj = sobj[sobj['flags'] == 0]

   print(len(sobj), 'transient candidates found in difference image')

The transient candidates detected by such a routine will require some filtering in order to reject the subtraction artefacts, probably filter out known variable stars and maybe catalogued stars, also filter out known minor planets, etc. We have a convenience high-level routine :func:`stdpipe.pipeline.filter_transient_candidates` that uses a number of lower-level ones (:func:`stdpipe.catalogs.xmatch_objects`, :func:`stdpipe.catalogs.xmatch_skybot`, :func:`stdpipe.catalogs.xmatch_ned`). High-level routine is able to either filter out the "bad" objects from the list of candidates, or just mark the ones that should be rejected for various reasons (if :code:`remove=False`).

.. code-block:: python

   # Filter out all candidates that match the objects from Pan-STARRS DR1 or AAVSO VSX catalogues
   # Also filter out all Solar System minor planets known to IMCCE SkyBoT service
   # Also, filter out all entries having flags 0x100 or 0x200 (thus, corresponding to objects
   # with masked pixels in their footprints
   candidates = pipeline.filter_transient_candidates(sobj, sr=2/3600,
                flagged=True, vizier=['ps1', 'vsx'], skybot=True, time=time, verbose=True)

   # ...or, we may do the same without removing the candidates:
   candidates = pipeline.filter_transient_candidates(sobj, sr=2/3600, remove=False,
                flagged=True, vizier=['ps1', 'vsx'], skybot=True, time=time, verbose=True)
   # and remove them later manually
   candidates = candidates[candidates['candidate_good']==True]

   # and now just print all the remaining candidates
   for i,cand in enumerate(candidates):
        print('Candidate %d with mag = %.2f +/- %.2f at x/y = %.1f %.1d and RA/Dec = %.4f %.4f' %
                (i, cand['mag_calib'], cand['mag_calib_err'], cand['x'], cand['y'], cand['ra'], cand['dec']))

.. autofunction:: stdpipe.pipeline.filter_transient_candidates
   :noindex:
