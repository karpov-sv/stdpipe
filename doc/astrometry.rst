Astrometric calibration
=======================

Astrometric solution is necessary for any reasonable analysis of astronomical imaging data. It is usually being represented as an `Astropy World Coordinate System <https://docs.astropy.org/en/stable/wcs/index.html>`_ objects that may be directly loaded from FITS headers - if they contain it, of course!

.. code-block:: python

   # Load FITS header from file
   header = fits.getheader(filename)

   # Construct WCS structure from this header
   wcs = WCS(header)

We have some simple convenience routines that may be used to quickly extract some information from it - like image center and size (:func:`stdpipe.astrometry.get_frame_center`) or pixel scale (:func:`stdpipe.astrometry.get_pixscale`).

.. code-block:: python

   # Get the center position, size and pixel scale for the image from WCS
   center_ra,center_dec,center_sr = astrometry.get_frame_center(wcs=wcs,
                width=image.shape[1], height=image.shape[0])
   pixscale = astrometry.get_pixscale(wcs=wcs)

   print('Frame center is %.2f %.2f radius %.2f deg, %.2f arcsec/pixel' %
                (center_ra, center_dec, center_sr, pixscale*3600))


Initial astrometric solution (blind solving)
--------------------------------------------

If your FITS header does not have any initial astrometric solution, you may derive it using `Astrometry.Net <https://nova.astrometry.net>`_ blind matching algorithm. We have two routines for working with it.

First one, :func:`stdpipe.astrometry.blind_match_astrometrynet`, directly uses their on-line service through `Astroquery <https://astroquery.readthedocs.io/en/latest/astrometry_net/astrometry_net.html>`_ library and requires an API key to be present in your :file:`~/.astropy/config/astroquery.cfg` config file

.. note::
   To get API key, you have to sign into your account at https://nova.astrometry.net and then generate the key on your profile page. Then, you should add it to your file :file:`~/.astropy/config/astroquery.cfg` by adding there

   .. code-block:: ini

      [astrometry_net]
      # The Astrometry.net API key
      api_key = 'your-api-key-goes-here'

   If API key is not set, the function call will produce a warning message and fail.

   You may also directly provide API key to the function call using `api_key` argument.

The function operates on the list of detected objects, so in principle network stress from its use is quite small (it will not try to upload your images to remote servers!). To speed up the solution, you may directly specify approximate center coordinates of the field center, as well as lower and upper bounds on the solution pixel scale.

Second routine, :func:`stdpipe.astrometry.blind_match_objects`, requires `local installation of Astrometry.Net binaries and index files <http://astrometry.net/use.html>`_. The routine itself has mostly the same signature as previous one, with some additional options to specify the configuration of the solver in finer details.

Example of using the code for solving the astrometry if not set in the header:

.. code-block:: python

   if wcs is None or not wcs.is_celestial:
       print('WCS is not set, blind solving for it...')

       # Try to guess rough frame center
       ra0 = header.get('RA')
       dec0 = header.get('DEC')
       sr0 = 1.0 # Should be enough?..

       # Should be sufficient for most images
       pixscale_low = 0.3
       pixscale_upp = 3

       # We will use no more than 500 brightest objects with S/N>10 to solve
       wcs = astrometry.blind_match_astrometrynet(obj[:500], center_ra=ra0, center_dec=dec0,
                radius=sr0, scale_lower=pixscale_low, scale_upper=pixscale_upp, sn=10)
       # .. or the same using local solver with custom config
       wcs = astrometry.blind_match_objects(obj[:500], center_ra=ra0, center_dec=dec0, radius=sr0,
                scale_lower=pixscale_low, scale_upper=pixscale_upp, sn=10, verbose=True,
                _exe='/home/karpov/astrometry/bin/solve-field',
                config='/home/karpov/astrometry/etc/astrometry-2mass.cfg')

       if wcs is not None and wcs.is_celestial:
           print('Blind solving succeeded!')

.. autofunction:: stdpipe.astrometry.blind_match_astrometrynet
   :noindex:

.. autofunction:: stdpipe.astrometry.blind_match_objects
   :noindex:

Astrometric refinement
----------------------

Existing approximate astrometric solution may be further improved to better represent image distortions. *STDPipe* provides several methods, all accessible through the higher-level wrapper :func:`stdpipe.pipeline.refine_astrometry`:

- ``method='quadhash'`` (default, recommended) - pure Python quad-hash pattern matching via :func:`stdpipe.astrometry_quad.refine_wcs_quadhash`
- ``method='scamp'`` - external `SCAMP <https://github.com/astromatic/scamp>`_ binary via :func:`stdpipe.astrometry.refine_wcs_scamp`
- ``method='astropy'`` - simple SIP fitting via AstroPy's ``fit_wcs_from_points``, through :func:`stdpipe.astrometry.refine_wcs_simple`
- ``method='astrometrynet'`` - SIP fitting via Astrometry.Net's ``fit-wcs`` binary, through :func:`stdpipe.astrometry.refine_wcs_simple`

.. autofunction:: stdpipe.pipeline.refine_astrometry
   :noindex:


Quad-hash refinement (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default and recommended method is quad-hash pattern matching, implemented in :func:`stdpipe.astrometry_quad.refine_wcs_quadhash`. It is a pure Python implementation with no external dependencies (only numpy, scipy, astropy).

Key features:

- **Pure Python** - no external binaries needed
- **Robust pattern matching** - uses geometric hashing of star quadrilaterals for reliable matching even with large initial WCS errors
- **High accuracy** - typically 2-7x more accurate than SCAMP, with sub-arcsecond residuals
- **SIP distortion fitting** - supports polynomial distortion orders 1-3
- **Projection-independent** - works with any WCS projection (TAN, ZEA, SIN, STG, ARC, etc.), not just TAN
- **ZPN projection for wide fields** - for images with FoV > 5°, the zenithal polynomial (ZPN) projection with PV radial terms plus optional SIP corrections gives lower residuals than TAN-SIP
- **Iterative refinement** - affine re-matching and progressive sigma-clipping for robust outlier rejection

.. code-block:: python

   # Default method - quad-hash refinement
   wcs = pipeline.refine_astrometry(obj, cat, 5*pixscale, wcs=wcs,
                cat_col_mag='rmag', verbose=True)

   # Equivalent explicit call
   from stdpipe.astrometry_quad import refine_wcs_quadhash
   wcs = refine_wcs_quadhash(obj, cat, wcs=wcs, sr=10/3600,
                order=2, sn=5, verbose=True)

   # Wide-field images (FoV > 5 degrees): use ZPN projection
   wcs = refine_wcs_quadhash(obj, cat, wcs=wcs, sr=30/3600,
                order=2, projection='ZPN', pv_deg=5,
                sn=5, verbose=True)

   if wcs is not None:
       # Update WCS info in the header
       astrometry.clear_wcs(header, remove_comments=True,
                remove_underscored=True, remove_history=True)
       header.update(wcs.to_header(relax=True))

.. autofunction:: stdpipe.astrometry_quad.refine_wcs_quadhash
   :noindex:


SCAMP refinement
^^^^^^^^^^^^^^^^

Alternatively, you may use `SCAMP <https://github.com/astromatic/scamp>`_ through :func:`stdpipe.astrometry.refine_wcs_scamp`. This requires external SCAMP binary to be installed.

.. attention::
   SCAMP, as well as SExtractor and SWarp, uses older and non-standard way of describing the distortions in the image - PV polynomials, while most of other softwares nowadays - including Astrometry.Net - prefer (standard) SIP polynomials. Python WCS loads both nicely, but there is no way to specify which one you wish to save back! Thus, converting one to another is not transparent, and you should also be aware which one you use! E.g. feeding SCAMP with initial WCS containing SIP polynomials will probably work, and it will return PV polynomial back that may be used for SWarping it. But directly running SWarp on Astrometry.Net - generated solutions will give you wrong results!

   You may read the discussion of the problem e.g. there - https://github.com/evertrol/sippv

.. code-block:: python

   # Use SCAMP for astrometric refinement
   wcs = pipeline.refine_astrometry(obj, cat, 5*pixscale, wcs=wcs,
                method='scamp', cat_col_mag='rmag', verbose=True)

.. autofunction:: stdpipe.astrometry.refine_wcs_scamp
   :noindex:


Simple SIP fitting (astropy / astrometrynet)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`stdpipe.astrometry.refine_wcs_simple` provides basic SIP distortion fitting using either AstroPy's ``fit_wcs_from_points`` (``method='astropy'``) or Astrometry.Net's ``fit-wcs`` binary (``method='astrometrynet'``). These methods perform a straightforward polynomial fit to pre-matched object/catalogue positions without the robust pattern matching of quad-hash or the full astrometric model of SCAMP.

.. code-block:: python

   # AstroPy-based SIP fitting
   wcs = pipeline.refine_astrometry(obj, cat, 5*pixscale, wcs=wcs,
                method='astropy', order=2, cat_col_mag='rmag', verbose=True)

   # Using Astrometry.Net's fit-wcs binary (requires local installation)
   wcs = pipeline.refine_astrometry(obj, cat, 5*pixscale, wcs=wcs,
                method='astrometrynet', order=2, cat_col_mag='rmag', verbose=True)

.. autofunction:: stdpipe.astrometry.refine_wcs_simple
   :noindex:


Residual-field correction
-------------------------

After the WCS fit, matched ``obj``--``cat`` pairs typically still show systematic per-position residuals: a few hundredths of a pixel from un-modelled distortions, a residual tilt the polynomial could not absorb, or PSF-position biases that the WCS solver cannot see. *STDPipe* can fit a smooth two-component ``(dx, dy)`` correction field on top of the WCS and either return it as a callable or apply it in place.

The primitive lives in :mod:`stdpipe.astrometry`:

- :func:`~stdpipe.astrometry.fit_astrometric_residuals` -- raw API: take observed and WCS-predicted pixel positions of matched sources, return a callable ``correct(x, y) -> (x_corrected, y_corrected)``.
- :func:`~stdpipe.astrometry.refine_positions_from_catalog` -- convenience that takes a match dict (the output of :func:`~stdpipe.pipeline.calibrate_photometry` or :func:`~stdpipe.photometry.match`), pulls out the matched positions, and forwards to the primitive. Returns ``(correct, info)`` where ``info`` carries pre/post-correction median and 90th-percentile residual magnitudes.

Both default to a fast bilinear-grid backend (``backend='grid'``) suitable for dense forced-photometry workloads (~600--1000x faster at prediction than LOESS); pass ``backend='loess'`` for higher-quality smoothing when prediction is needed at modest numbers of points. The underlying smoother is :func:`stdpipe.smoothing.fit_vector_field_2d`.

.. code-block:: python

   # Build the correction from an already-matched obj/cat pair
   from stdpipe import astrometry, pipeline

   m = pipeline.calibrate_photometry(obj, cat, sr=2/3600, wcs=wcs,
                                     cat_col_mag='rmag', verbose=True)
   correct, info = astrometry.refine_positions_from_catalog(
       m, obj, cat, wcs,
       image_shape=image.shape,
       grid_shape=(8, 6),       # binned-grid resolution
   )
   print('median |Δ| {:.3f} → {:.3f} px, q90 {:.3f} → {:.3f} px'.format(
       info['raw_median_dr_pix'], info['corrected_median_dr_pix'],
       info['raw_q90_dr_pix'], info['corrected_q90_dr_pix']))

   # Use the correction to refine catalogue-projected positions e.g. for
   # forced-position aperture photometry of every catalogue source.
   x_pred, y_pred = wcs.all_world2pix(cat['ra'], cat['dec'], 0)
   x_corr, y_corr = correct(x_pred, y_pred)


.. autofunction:: stdpipe.astrometry.fit_astrometric_residuals
   :noindex:

.. autofunction:: stdpipe.astrometry.refine_positions_from_catalog
   :noindex:


Pipeline integration
^^^^^^^^^^^^^^^^^^^^

:func:`~stdpipe.pipeline.refine_astrometry` accepts opt-in ``refine_residual_field=True`` and ``residual_field_kwargs=`` parameters. When enabled, after the WCS fit the function re-matches ``obj`` to ``cat`` positionally via :func:`~stdpipe.astrometry.spherical_match`, fits the residual field, and -- when ``update=True`` -- subtracts the smooth correction from ``obj['x']``/``obj['y']`` in place before re-deriving ``obj['ra']``/``obj['dec']`` from the refined positions. ``obj['x']``/``['y']`` then lie on the WCS-consistent grid and downstream code can work with the WCS alone, without carrying the correction model.

.. code-block:: python

   wcs = pipeline.refine_astrometry(
       obj, cat, sr=2/3600, wcs=wcs, order=2, cat_col_mag='rmag',
       refine_residual_field=True,
       residual_field_kwargs=dict(
           image_shape=image.shape,
           grid_shape=(8, 6),
           min_per_cell=6, smooth_sigma=1.0,
       ),
       verbose=True,
   )

A short diagnostic line is logged with the number of matches used and the median/90th-percentile residual magnitudes before and after the correction. The block silently skips when fewer than 50 positional matches survive, leaving ``obj`` untouched and the WCS unchanged.
