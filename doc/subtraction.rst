Image subtraction
=================

Any attempt of detection of variable or transient sources in a reasonably dense field, or on typical background for extragalactic transients (read - on the outskirts of galaxies, often bright enough) requires image subtraction methods.

Template images
---------------

Image subtraction requires template image that has to be astrometrically aligned with your science one. If you have your own set of deep enough images you may construct the template by the methods described in the :ref:`Stacking the images` section above.

*STDPipe* also has a couple of functions that may help you downloading template images from publicly available archives. All of them will deliver *ready to use* image that is already projected on the pixel grid of your science frame and has the same shape.

.. code-block:: python

   # Get r band image from PanSTARRS with science image original resolution and orientation
   tmpl = templates.get_hips_image('PanSTARRS/DR1/r', wcs=wcs,
                width=image.shape[1], height=image.shape[0], get_header=False, verbose=True)

   # Now mask some brighter stars in the template as we know they are saturated in Pan-STARRS
   tmask = templates.mask_template(tmpl, cat, cat_col_mag='rmag', cat_saturation_mag=15,
                wcs=wcs, dilate=3, verbose=True)

   # And now the same - using original Pan-STARRS images and masks
   tmpl,tmask = templates.get_ps1_image_and_mask('r', wcs=wcs, width=, height=, verbose=True)


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


Using original Pan-STARRS or Legacy Survey images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*STDPipe* is also able to directly download the images from `Pan-STARRS <https://outerspace.stsci.edu/display/PANSTARRS/Pan-STARRS1+data+archive+home+page>`_
or `Legacy Survey <https://www.legacysurvey.org>`_ image archives, mosaic them and project onto requested pixel grid in order to produce the template. It is also able to simultaneously acquite the `mask` image so that you may properly exclude unreliable template pixels from the analysis.

.. autofunction:: stdpipe.templates.get_survey_image
   :noindex:

.. autofunction:: stdpipe.templates.get_survey_image_and_mask
   :noindex:

.. autofunction:: stdpipe.templates.get_ps1_image
   :noindex:

.. autofunction:: stdpipe.templates.get_ps1_image_and_mask
   :noindex:


Running image subtraction
-------------------------

*STDPipe* provides several image subtraction methods:

- **SFFT** — built-in spatially varying kernel fitting via :func:`stdpipe.subtraction.run_sfft` (no external dependencies)
- **HOTPANTS** — external `HOTPANTS <https://github.com/acbecker/hotpants>`_ code via :func:`stdpipe.subtraction.run_hotpants` (requires separate installation)
- **ZOGY** — built-in implementation of the Zackay–Ofek–Gal-Yam algorithm via :func:`stdpipe.subtraction.run_zogy`

All three wrappers follow the same interface conventions: they accept science and template images as NumPy arrays, optional masks and noise maps, and return the difference image along with optional noise maps, noise-normalized images, and convolved templates.


SFFT subtraction
^^^^^^^^^^^^^^^^

:func:`stdpipe.subtraction.run_sfft` performs image subtraction using the SFFT algorithm (Hu et al. 2022, ApJ, 936, 157). It solves for a spatially varying convolution kernel and differential background in a single global least-squares problem, without requiring any external tools.

The kernel at each pixel is modelled as a delta-function basis with polynomial spatial variation. A soft constraint enforces that the kernel sum (flux scale) varies smoothly as a low-order polynomial, accounting for differences in photometric calibration between science and template.

.. code-block:: python

   # Basic subtraction with automatic noise estimation
   diff = subtraction.run_sfft(image, tmpl,
                mask=mask, template_mask=tmask,
                err=True, template_err=True,
                image_gain=gain, template_gain=1e6,
                verbose=True)

   # Get all output planes (like HOTPANTS)
   diff, conv, noise, scaled, sfft_result = subtraction.run_sfft(image, tmpl,
                mask=mask, template_mask=tmask,
                err=True, template_err=True,
                image_gain=gain, template_gain=1e6,
                get_convolved=True, get_noise=True,
                get_scaled=True, get_kernel=True,
                verbose=True)

   # Now we have:
   # - `diff` for the difference image
   # - `conv` for the template convolved to match the science image
   # - `noise` for the per-pixel noise map of the difference image
   # - `scaled` for the noise-normalized difference image (diff / noise)
   # - `sfft_result` for the full SFFTResult with kernel coefficients and metadata

   # Inspect the spatially varying kernel at a specific position
   from stdpipe import sfft
   kernel_center = sfft.evaluate_kernel_at(sfft_result, 500, 500, image.shape)
   flux_scale = sfft.evaluate_flux_scale(sfft_result, 500, 500, image.shape)

.. autofunction:: stdpipe.subtraction.run_sfft
   :noindex:


HOTPANTS subtraction
^^^^^^^^^^^^^^^^^^^^

:func:`stdpipe.subtraction.run_hotpants` is a wrapper for the `HOTPANTS <https://github.com/acbecker/hotpants>`_ image subtraction code. It requires HOTPANTS to be installed separately (see :ref:`Installation`). We recommend checking the HOTPANTS documentation to better understand the concepts and options for it.

.. code-block:: python

   # Run the subtraction getting back all possible image planes, assuming the template
   # to be noise-less, and estimating image noise model from its statistics.
   # And also pre-flatting the images before subtraction to get rid
   # of possible background inhomogeneities.

   import photutils

   bg = photutils.Background2D(image, 128, mask=mask, exclude_percentile=30).background
   tbg = photutils.Background2D(tmpl, 128, mask=tmask, exclude_percentile=30).background

   diff,conv,sdiff,ediff = subtraction.run_hotpants(image-bg, tmpl-tbg,
                mask=mask, template_mask=tmask,
                get_convolved=True, get_scaled=True, get_noise=True,
                image_fwhm=fwhm, template_fwhm=1.5,
                image_gain=gain, template_gain=1e6,
                err=True, verbose=True)

   # Now we have:
   # - `diff` for the difference image
   # - `conv` for the template convolved to match the original image
   # - `sdiff` for noise-normalized difference image - ideal for quickly seeing significant deviations!
   # - `ediff` for the difference image noise model - you may use it to weight object detection to reject the subtraction artefacts e.g. around brighter objects

.. autofunction:: stdpipe.subtraction.run_hotpants
   :noindex:
