Image subtraction
=================

Any attempt of detection of variable or transient sources in a reasonably dense field, or on typical background for extragalactic transients (read - on the outskirts of galaxies, often bright enough) requires image subtraction methods.

Template images
---------------

Image subtraction requires template image that has to be astrometrically aligned with your science one. If you have your own set of deep enough images you may construct the template by the methods described in the :ref:`Stacking the images` section above.

*STDPipe* also has a couple of functions that may help you downloading template images from publicly available archives. All of them will deliver *ready to use* image that is already projected on the pixel grid of your science frame and has the same shape.

Using Hierarchical Progressive Survey (HiPS) images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First one, :func:`stdpipe.templates.get_hips_image`, acquires template images from any HiPS (Hierarchical Progressive Survey) formatted survey available on the net (see the `full list <https://aladin.u-strasbg.fr/hips/list>`_ - it is huge!). The routine uses CDS `hips2fits <http://alasky.u-strasbg.fr/hips-image-services/hips2fits>`_ service to do the job.

.. autofunction:: stdpipe.templates.get_hips_image
   :noindex:

.. attention::
   Depending on the survey, the results may wary. Be aware that masked pixels may either be represented as `NaN` in the image, or be silently interpolated, that may impact the analysis of these images.

   Also, right now Pan-STARRS images are in the process of re-uploading to CDS that will supposedly fix the problems due to its orininal non-lineary flux scaling. So expect the routine to produce nonsensical output for it!

We have a convenience function that may help masking the pixels that are most probably unreliable in a given HiPS image (e.g. above survey saturation theshold) if they are somehow not masked (not set to `NaN`).

.. autofunction:: stdpipe.templates.mask_template
   :noindex:

Using original Pan-STARRS images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*STDPipe* is also able to directly download the images from `Pan-STARRS image archive <https://outerspace.stsci.edu/display/PANSTARRS/Pan-STARRS1+data+archive+home+page>`_, mosaic them and project onto requested pixel grid in order to produce the template. It is also able to simultaneously acquite the `mask` image so that you may properly exclude unreliable template pixels from the analysis.

.. autofunction:: stdpipe.templates.get_ps1_image
   :noindex:

.. autofunction:: stdpipe.templates.get_ps1_image_and_mask
   :noindex:


Running image subtraction
-------------------------

*STDPipe* has some basic support for image subtraction through the interface to `HOTPANTS <https://github.com/acbecker/hotpants>`_ image subtraction code that is implemented in :func:`stdpipe.subtraction.run_hotpants`.


.. autofunction:: stdpipe.subtraction.run_hotpants
   :noindex:
