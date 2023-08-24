Pre-processing the data
=======================

The codes in *STDPipe* expect as an input the *science-ready* images, cleaned as much as possible from instrumental signatures and imaging artefacts. In practice, it means that the image should be

-  bias and dark subtracted
-  flat-fielded.

Also, the artefacts such as saturated stars, bleeding charges, cosmic ray hits etc have to be masked.

All these tasks are outside of *STDPipe* per se, as they are highly instrument and site specific.
On the other hand, they may usually be performed easily using standard Python/NumPy/AstroPy routines and libraries like `Astro-SCRAPPY <https://github.com/astropy/astroscrappy>`_ etc.

E.g. to pre-process the raw image using pre-computed master dark and flat, mask some common problems, and cleanup the cosmic rays you may do something like that:

.. code-block:: python

   image = fits.getdata(filename).astype(np.double)
   header = fits.getheader(filename)

   dark = fits.getdata(darkname)
   flat = fits.getdata(flatname)

   image -= dark
   image *= np.median(flat)/flat

   saturation = header.get('SATURATE') or 50000 # Guess saturation level from FITS header

   mask = np.isnan(image) # mask NaNs in the input image
   mask |= image > saturation # mask saturated pixels
   mask |= flat < 0.5*np.median(flat) # mask underilluminated/vignetted regions

   from astropy.stats import mad_std
   mask |= dark > np.median(dark) + 10.0*mad_std(dark) # mask hotter pixels

   gain = header.get('GAIN') or 1.0 # Guess gain from FITS header

   import astroscrappy
   # mask cosmic rays using LACosmic algorithm
   cmask,cimage = astroscrappy.detect_cosmics(image, mask, gain=gain, verbose=True)
   mask |= cmask


We have a simple routine that implements these steps, :func:`stdpipe.pipeline.make_mask`, which is documented below.

.. autofunction:: stdpipe.pipeline.make_mask
   :noindex:


Stacking the images
-------------------

You may want to stack/coadd or mosaic some images before processing them. While there are dedicated large-scape packages like `Montage <http://montage.ipac.caltech.edu>`_ that handle it *properly*, it still may be done manually with relatively little efforts using e.g. Python `reproject <https://github.com/astropy/reproject>`_ package.

You may check `simple example notebook <https://github.com/karpov-sv/stdpipe/blob/master/notebooks/image_stacking.ipynb>`_
that shows how to do it.

.. attention::
   The stacking modify the statistical properties of resulting image! The reasons are both averaging (or especially median averaging!) of the images that modify effective gain value (typically increasing it by the factor equal to number of averaged images), and pixel interpolation when re-projecting the images onto the same pixel grid.

*STDPipe* also contains a simple wrapper for `SWarp <https://github.com/astromatic/swarp>`_ re-projection code, :func:`stdpipe.templates.reproject_swarp`, that is implemented to resemble the calling conventions of `reproject <https://github.com/astropy/reproject>`_ package routines as much as possible - i.e. allows directly stacking the image files without loading them to memory first:

.. autofunction:: stdpipe.templates.reproject_swarp
   :noindex:
