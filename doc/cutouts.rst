Image cutouts
=============

Image cutouts, or *postage stamps*, are useful for quick visual characterization of the detected transients or artefacts. We have a set of functions inside :mod:`stdpipe.cutouts` sub-module that ease these tasks.

They all work with the cutout structure returned by :func:`stdpipe.cutouts.get_cutout` that contains the stamps of requested size around the object from various image planes (science image, template, background map, mask, footprint map, ...), as well as some meta-information.

.. code-block:: python

   # Create and display the cutouts from an image and its mask

   for i,cand in enumerate(candidates):
       # Create the cutout from image based on the candidate
       cutout = cutouts.get_cutout(image, cand, 20, mask=mask, header=header)

       # We may directly download the template image for this cutout
       # from HiPS server - same scale and orientation, but different PSF shape!..
       cutout['template'] = templates.get_hips_image('PanSTARRS/DR1/r',
                header=cutout['header'], get_header=False)

       # We do not have difference image, so it will only display original one, template and mask
       plots.plot_cutout(cutout, planes=['image', 'template', 'mask'],
                qq=[0.5, 99.9], stretch='linear')
       plt.show()

.. code-block:: python

   # Create and display the cutouts from a full set of image subtraction results

   for i,cand in enumerate(candidates):
       print('Candidate %d with mag = %.2f +/- %.2f at x/y = %.1f %.1d and RA/Dec = %.4f %.4f' %
                (i, cand['mag_calib'], cand['mag_calib_err'], cand['x'], cand['y'], cand['ra'], cand['dec']))

       cutout = cutouts.get_cutout(image, cand, 20, mask=mask|tmask, diff=diff, template=tmpl,
                convolved=conv, err=ediff, header=header, filename=filename, time=time)

       plots.plot_cutout(cutout, ['image', 'template', 'convolved', 'diff', 'mask'],
                qq=[0.5, 99.5], stretch='linear')

       plt.show()

       # Also, store the cutout!
       cutouts.write_cutout(cutout, 'candidates/cutout_%03d.fits' % i)

.. autofunction:: stdpipe.cutouts.get_cutout
   :noindex:


Saving and loading the cutouts
------------------------------

Cutouts may be easily written to multi-extension FITS images, and restored from them later using the following functions:

.. autofunction:: stdpipe.cutouts.write_cutout
   :noindex:

.. autofunction:: stdpipe.cutouts.load_cutout
   :noindex:


Plotting the cutouts
--------------------

There is a dedicated plotting routine that helps quickly visualize the information contained in it, including its different image planes - :func:`stdpipe.plots.plot_cutout`

.. autofunction:: stdpipe.plots.plot_cutout
   :noindex:


Cutout-level rejection of subtraction artefacts
-----------------------------------------------

The difference image often contains the characteristic artefacts ("dipoles") due to slight sub-pixel displacement of the object between the science image and the template (e.g. due to imperfect astrometric alignment, or due to large proper motion for the templates acquired long ago). Also, the intensity of the object may also be slightly different on the science image - e.g. due to slightly different photometric band - that leads to unshifted positive or negative differences. In principle, such cases may be automatically detected and - if appropriate - rejected by a simple adjustment procedure based solely on the information we already have inside typical cutout. The adjustment here means optimizing slight (sub-pixel) shifts between the image and (convolved) template, as well as slight scaling of the template, in order to minimize the residuals. We have a dedicated routine, :func:`stdpipe.cutouts.adjust_cutout`, that tries to perform such optimization, and may be used inside the pipelines like in the following example:

.. code-block:: python

   for i,cand in enumerate(candidates):
       print('Candidate %d with mag = %.2f +/- %.2f at x/y = %.1f %.1d and RA/Dec = %.4f %.4f' %
                (i, cand['mag_calib'], cand['mag_calib_err'], cand['x'], cand['y'], cand['ra'], cand['dec']))

       # Produce the cutout with all necessary planes
       cutout = cutouts.get_cutout(image, cand, 20, mask=mask|tmask, diff=diff, template=tmpl,
                convolved=conv, err=ediff, header=header, filename=filename, time=time)

       # Try to adjust the position and scale of the cutout to see whether it disappears or not
       # Here we use only inner 2*fwhm x 2*fwhm part for fitting
       # and limit possible shifts and scale to 1 pixel and 2.0, correspondingly
       if cutouts.adjust_cutout(cutout, inner=2*fwhm, max_shift=1, max_scale=2.0,
                fit_bg=True, normalize=False, verbose=True):
           # The optimization converged - now we may check its actual adjustment
           # and decide whether the transient disappeared or not
           # E.g. we see whether final chi2 is at least 3 times smaller than original one
           # and its corresponding p-value is worse than 1e-5
           # and the fitted scale is between 0.8 and 1.2
           if ((cutout['meta']['adjust_pval'] > 1e-5 or
                cutout['meta']['adjust_chi2'] < 0.3*cutout['meta']['adjust_chi2_0']) and
               cutout['meta']['adjust_scale'] > 0.8 and cutout['meta']['adjust_scale'] < 1.2):
               # Well, it really "disappeared"
               print("The candidate is no more significant"
               cutout['meta']['adjusted'] = True

       # We may now plot the results, including the results of the adjustment routine
       # It will be shown in `adjusted` plane, if the optimization converged
       plots.plot_cutout(cutout, ['image', 'template', 'convolved', 'diff', 'adjusted', 'mask'],
                qq=[0.5, 99.5], stretch='linear')



.. autofunction:: stdpipe.cutouts.adjust_cutout
   :noindex:
