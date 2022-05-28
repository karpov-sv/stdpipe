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
   center_ra,center_dec,center_sr = astrometry.get_frame_center(wcs=wcs, width=image.shape[1], height=image.shape[0])
   pixscale = astrometry.get_pixscale(wcs=wcs)

   print('Frame center is %.2f %.2f radius %.2f deg, %.2f arcsec/pixel' % (center_ra, center_dec, center_sr, pixscale*3600))

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
       wcs = astrometry.blind_match_astrometrynet(obj[:500], center_ra=ra0, center_dec=dec0, radius=sr0, scale_lower=pixscale_low, scale_upper=pixscale_upp, sn=10)
       # .. or the same using local solver with custom config
       wcs = astrometry.blind_match_objects(obj[:500], center_ra=ra0, center_dec=dec0, radius=sr0, scale_lower=pixscale_low, scale_upper=pixscale_upp, sn=10, verbose=True, _exe='/home/karpov/astrometry/bin/solve-field', config='/home/karpov/astrometry/etc/astrometry-2mass.cfg')

       if wcs is not None and wcs.is_celestial:
           print('Blind solving succeeded!')

.. autofunction:: stdpipe.astrometry.blind_match_astrometrynet
   :noindex:

.. autofunction:: stdpipe.astrometry.blind_match_objects
   :noindex:

Astrometric refinement
----------------------

Existing approximate astrometric solution may be further improved to better represent image distortions. It may be done using `SCAMP <https://github.com/astromatic/scamp>`_ code - we have a routine :func:`stdpipe.astrometry.refine_wcs_scamp` that conveniently wraps it, hiding most of trickier details of running it.

.. attention::
   SCAMP, as well as SExtractor and SWarp, uses older and non-standard way of describing the distortions in the image - PV polynomials, while most of other softwares nowadays - including Astrometry.Net - prefer (standard) SIP polynomials. Python WCS loads both nicely, but there is no way to specify which one you wish to save back! Thus, converting one to another is not transparent, and you should also be aware which one you use! E.g. feeding SCAMP with initial WCS containing SIP polynomials will probably work, and it will return PV polynomial back that may be used for SWarping it. But directly running SWarp on Astrometry.Net - generated solutions will give you wrong results!

   You may read the discussion of the problem e.g. there - https://github.com/evertrol/sippv

We also have another, less reliable and less tested routine :func:`stdpipe.astrometry.refine_wcs` that will operate with SIP distortions, and do so either in pure Python, or using `fit-wcs` executable from Astrometry.Net installation. We also have a higher-level routine that uniformly wraps all these functions - :func:`stdpipe.pipeline.refine_astrometry`.

Example:

.. code-block:: python

   # Let's use SCAMP for astrometric refinement.
   wcs = pipeline.refine_astrometry(obj, cat, 5*pixscale, wcs=wcs, method='scamp', cat_col_mag='rmag', verbose=True)

   if wcs is not None:
       # Update WCS info in the header
       astrometry.clear_wcs(header, remove_comments=True, remove_underscored=True, remove_history=True)
       header.update(wcs.to_header(relax=True))

.. autofunction:: stdpipe.astrometry.refine_wcs_scamp
   :noindex:

.. autofunction:: stdpipe.pipeline.refine_astrometry
   :noindex:
