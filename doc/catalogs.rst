Reference catalogues
====================

In order to perform :doc:`astrometric <astrometry>` and :doc:`photometric <photometry>` calibrations, you will need a reference catalogue. For *STDPipe*, any table that contains sky positions of the stars and their magnitudes will work - so you may either construct it yourself, load from file or database, or download from Internet.

For the latter, we have a convenience routine :func:`stdpipe.catalogs.get_cat_vizier` that helps accessing any catalogue available in `Vizier <https://vizier.cds.unistra.fr>`__.

For some of them we have a special support - we automatically augment them with magnitudes in photometric bands not originally present there. To do so, we use the following sources of photometric transformations:

-  for Pan-STARRS to Johnson-Cousins magnitudes, we use conversion coefficients from https://arxiv.org/pdf/1706.06147.pdf

-  for Gaia DR2, we use some self-made conversion coefficients derived by cross-matching with Stetson standards, along with linearity correction described at https://www.cosmos.esa.int/web/gaia/dr2-known-issues#PhotometrySystematicEffectsAndResponseCurves

The following example shows how to get a reference catalogue for a given image

.. code-block:: python

   # Load initial WCS from FITS header
   wcs = WCS(header)

   # Get the center position, size and pixel scale for the image from WCS
   center_ra,center_dec,center_sr = astrometry.get_frame_center(wcs=wcs,
                width=image.shape[1], height=image.shape[0])
   pixscale = astrometry.get_pixscale(wcs=wcs)

   print('Frame center is %.2f %.2f radius %.2f deg, %.2f arcsec/pixel' %
                (center_ra, center_dec, center_sr, pixscale*3600))

   # Load Pan-STARRS stars for this region brighter than r=18.0 mag
   cat = catalogs.get_cat_vizier(center_ra, center_dec, center_sr, 'ps1',
                filters={'rmag':'<18'})
   print(len(cat), 'catalogue stars')

.. attention::

   These catalogue-access functions all use `Astroquery <https://astroquery.readthedocs.io/en/latest/>`__ package to do the actual work. It caches the results (so that consecutive queries to the same region of the same catalogue do not need to re-download the data) inside your home directory - in the :file:`~/.astropy/cache/astropy/` directory. There is no built-in way to limit its size yet - so you will need to monitor its disk space usage, and probably clean it up manually from time to time!


.. autofunction:: stdpipe.catalogs.get_cat_vizier
   :noindex:

Cross-matching with Vizier catalogues
-------------------------------------

*STDPipe* also has a couple of convenience routines for cross-matching object lists with Vizier catalogues that use `CDS XMatch service <http://cdsxmatch.u-strasbg.fr>`__, as well as with Solar system objects using `IMCCE SkyBoT service <http://vo.imcce.fr/webservices/skybot/>`__. Both are thin wrappers around corresponding routines from `Astroquery <https://astroquery.readthedocs.io/en/latest/index.html>`__ package.

.. autofunction:: stdpipe.catalogs.xmatch_objects
   :noindex:

.. autofunction:: stdpipe.catalogs.xmatch_skybot
   :noindex:
