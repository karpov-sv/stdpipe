Photometric calibration
=======================

Photometric calibration is performed by positionally matching detected objects with catalogue stars, and then building the photometric model for their instrumental magnitudes.

The model includes:

-  catalogue magnitude
-  (optionally spatially varying) zero point
-  (optionally) catalogue color
-  (very optionally) additive flux term, corresponding e.g. to biased background estimation
-  (very very optionally) intrinsic scatter on top of measurement errors

and its fitting is implemented in :func:`stdpipe.photometry.match` function, and wrapped into a bit higher-level and easier to use :func:`stdpipe.pipeline.calibrate_photometry`. The routines perform an iterative fitting with rejection of pairs deviating too much (more than `threshold` sigmas) from the model that is defined like that:

.. math::
   {\rm Catalogue} = {\rm Instrumental} + C\cdot{\rm color} + ZP(x, y, {\rm spatial_order}) + {\rm additive} + {\rm error}

where

*  :math:`{\rm Instrumental}=-2.5\log_{10}({\rm ADU})` is instrumental magnitude,
*  :math:`{\rm Catalogue}` is catalogue magnitude, e.g. :math:`{\rm V}`, and
*  :math:`{\rm color}` is catalogue color, e.g. :math:`{\rm B-V}`.

This is equivalent to the instrumenmtal photometric system defined as :math:`{\rm V} - C\cdot({\rm B-V})`.

Zero point :math:`ZP` is a spatially varying polynomial with the degree controlled by `spatial_order` parameter (:code:`spatial_order=0` for a constant zero point)

Additive flux term is defined by linearizing the additional flux in every photometric aperture (e.g. due to incorrect background level determination) and has a form

.. math::
   {\rm additive} = -2.5/\log(10)/10^{-0.4\cdot{\rm Instrumental}} \cdot {\rm bg_corr}(x, y, {\rm bg_order})

where :math:`{\rm bg_corr}` is a flux correction inside the aperture. This term is also spatially dependent, and is controlled by `bg_order` parameter. If :code:`bg_order=None`, the fitting for this term is disabled.

The calibration routine performs an iterative weighted linear least square (or robust if :code:`robust=True`) fitting with rejection of pairs deviating too much (more than `threshold` sigmas) from the model. Optional intrinsic scatter (specified through `max_intrinsic_rms` parameter) may also be fitted for, and may help accounting for the effects of e.g. multiplicative noise (flatfielding, subpixel sensitivity variations, etc).

.. attention::
   The calibration (or zero point, or systematic) error is currently estimated as a fitted model error, and is typically way too small. Most probably the method is wrong! So I welcome any input or discussions on this topic.

.. code-block:: python

   # Photometric calibration using 2 arcsec matching radius, r magnitude, g-r color and second order spatial variations
   m = pipeline.calibrate_photometry(obj, cat, sr=2/3600, cat_col_mag='rmag', cat_col_mag1='gmag', cat_col_mag2='rmag', max_intrinsic_rms=0.02, order=2, verbose=True)

   # The code above automatically augments the object list with calibrated magnitudes, but we may also do it manually
   obj['mag_calib'] = obj['mag'] + m['zero_fn'](obj['x'], obj['y'])
   obj['mag_calib_err'] = np.hypot(obj['magerr'], m['zero_fn'](obj['x'], obj['y'], get_err=True))

   # Now, if we happen to know object colors, we may also compute proper color-corrected magnitudes:
   obj['mag_calib_color'] = obj['mag_calib'] + obj['color']*m['color_term']

.. autofunction:: stdpipe.photometry.match
   :noindex:

.. autofunction:: stdpipe.pipeline.calibrate_photometry
   :noindex:

Plotting photometric match results
----------------------------------

*STDPipe* also includes a dedicated plotting routine :func:`stdpipe.plots.plot_photometric_match` that may be used to visually inspect various aspects of the photometric match.

.. code-block:: python

   # Photometric residuals as a function of catalogue magnitude
   plt.subplot(211)
   plots.plot_photometric_match(m)
   plt.ylim(-0.5, 0.5)

   # Photometric residuals as a function of catalogue color
   plt.subplot(212)
   plots.plot_photometric_match(m, mode='color')
   plt.ylim(-0.5, 0.5)
   plt.xlim(0.0, 1.5)
   plt.show()

   # Zero point (difference between catalogue and instrumental magnitudes for every star) map
   plots.plot_photometric_match(m, mode='zero', bins=6, show_dots=True, aspect='equal')
   plt.show()

   # Fitted zero point model map
   plots.plot_photometric_match(m, mode='model', bins=6, show_dots=True, aspect='equal')
   plt.show()

   # Residuals between empirical zero point and fitted model
   plots.plot_photometric_match(m, mode='residuals', bins=6, show_dots=True, aspect='equal')
   plt.show()

   # Astrometric displacement between objects and matched catalogue stars
   plots.plot_photometric_match(m, mode='dist', bins=6, show_dots=True, aspect='equal')
   plt.show()

.. autofunction:: stdpipe.plots.plot_photometric_match
   :noindex:
