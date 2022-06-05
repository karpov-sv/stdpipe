Convenience utilities
=====================


*STDPipe* also contains a number of convenience utilities that help solving some of maybe minor but still quite important problems you may encounter during data analysis and visualization.


Splitting the image into sub-images
-----------------------------------

Often you have quite large image, with significant variation of PSF, background etc over it. Such frames are not very well suited for image subtraction, as convolution kernel will become unstable and / or unable to properly match the resolutions of image and template.

So, for such images, it is better to split them into smaller pieces and do image subtraction on the latters. When splitting, however, you have to also split the mask, update WCS solution and FITS header, maybe select a subset of catalogue stars, etc etc. And we have a dedicated routine for doing just that!

.. code-block:: python

   # Crop the image (mask, header, WCS, object list and catalogue too) into 4 (2x2) pieces,
   # adding 10 pixels wide overlap between them.
   for i, x0, y0, image1, mask1, header1, wcs1, obj1, cat1 in pipeline.split_image(image, nx=2,
                mask=mask, header=header, wcs=wcs, obj=obj, cat=cat, overlap=10,
                get_index=True, get_origin=True, verbose=True):
       # We got a sub-image
       print('Subimage', i, 'has origin at', x0, y0, 'shape', image1.shape, 'and contains', len(obj1),
                'objects from original image')

       # Do something useful on the sub-images here!
       pass

.. autofunction:: stdpipe.pipeline.split_image
   :noindex:


Extracting the time from FITS header
------------------------------------

We have a routine that helps extracting the information on time of observations from FITS headers.

.. autofunction:: stdpipe.utils.get_obs_time
   :noindex:


Displaying the images
---------------------

We have a convenience image plotting function :func:`stdpipe.plots.imshow` what may be used as a drop-in replacement for :func:`matplotlib.pyplot.imshow`, but also supports percentile-based intensity min/max levels, various intensity stretching options (linear, asinh, log, etc), built-in colorbar that plays nicely with sub-plots, and an option to hide the axes around the image.

.. code-block:: python

   # Plot the image with asinh intensity scaling between [0.5, 99.5] quantiles, and a color bar
   plots.imshow(image, qq=[0.5, 99.5], stretch='asinh', show_colorbar=True)

.. autofunction:: stdpipe.plots.imshow
   :noindex:


Displaying 2d histograms of randomly distributed points
-------------------------------------------------------

We also have a convenience function for plotting the images of 2d histograms of irregularly spaced data points. It may shows various statistics of the point values - means, medians, standard deviations etc, as well as display the color bar for the values, do percentile-based intensity scaling, and overlay the positions of the data points onto the histogram.

.. code-block:: python

   # Show the map of FWHM of detected objects using 8x8 bins and overlaying the positions
   # of the objects. Also, force the image to have correct aspect ratio
   plots.binned_map(obj['x'], obj['y'], obj['fwhm'], cmap='hot', bins=8, aspect='equal',
                show_colorbar=True, show_dots=True, color='blue')


.. autofunction:: stdpipe.plots.binned_map
   :noindex:
