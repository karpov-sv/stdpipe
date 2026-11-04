# Changelog

All notable changes to STDPipe are documented in this file. Changes in
releases up to 0.4.0 are only available in the
[commit history](https://github.com/karpov-sv/stdpipe/commits/master).

## Unreleased

### Added

- `photometry.get_objects_sep()` accepts a `psf` model (PSFEx, `create_psf_model()`
  or `sep.PSF`) for PSF-matched detection on the `sep.psf_snr()` significance
  image, following spatial PSF variation; shapes, fluxes and FWHM are still
  measured on the image. The detection filter then matches the actual PSF,
  without tuning `r0`; with `minnthresh=0`, faint-star completeness equals that
  of a Gaussian kernel matched to the PSF width, and is far above that of the
  default `r0=0.5` kernel for typical PSFs. Requires sep-x.
- `get_objects_sep()` accepts `sn=None` (or 0) to disable the S/N cut.

## 0.5.0 - 2026-10-31

Also published as 0.4.2, which was yanked.

### Changed

- PSF fitters (`photometry_psf.measure_objects_psf()`,
  `photometry_measure.measure_objects_sep()` and
  `photometry_iraf.measure_objects_psf()`) now store fitted positions in
  `x`/`y` and keep the input ones in `x_orig`/`y_orig`, as centroiding does.
  The `x_psf`/`y_psf` columns are gone; `get_objects_sextractor()` still
  reports SExtractor's own `x_psf`/`y_psf`.
- `psf.create_psf_model()` no longer uses photutils `EPSFBuilder`. The new
  builder places star pixels at their true sub-pixel phases, iteratively
  recentres and renormalizes the model, rejects outliers per pixel, subtracts
  neighbours with the model itself, and sizes the stamp automatically from the
  stacked stellar profile. This removes a ~0.8% PSF flux bias and 2-6% flux
  losses for Moffat-like PSFs, and is several times faster. Resulting models
  differ slightly from previous versions.
- `measure_objects_psf()` default `maxiters` raised from 3 to 100 (it is a
  limit on function evaluations, and fits did not converge before), and
  non-convergence is now detected using the correct photutils flag bit.
- `measure_objects_sep()` now uses a 5-sigma clipped mean for the local
  background by default, for both aperture photometry (was unclipped median)
  and optimal extraction (was 3-sigma clipped mean).
- Legacy Survey templates are taken from DR11 instead of DR10. Previously
  cached DR10 bricks are reused as is, as their filenames do not encode the
  data release.
- `resolve.simbadResolve()` queries Simbad only (instead of Simbad, NED and
  VizieR), using the current Sesame URL.
- Requires `sep-x>=1.5.2`, which correctly resamples point-sampled
  supersampled PSF models (PSFEx, `create_psf_model()`).

### Added

- `create_psf_model(degree='auto')` selects the spatial polynomial degree of
  the PSF model by spatial cross-validation, up to `max_degree`.
- `order='auto'` for SIP order selection in `refine_wcs_quadhash()` and
  `refine_wcs_simple()`, as already supported by `refine_wcs_scamp()`.
- `bg_estimator` (`'mode'`, `'median'`, `'mean'`) with `clip_sigma`/`clip_iters`
  for the local background in `measure_objects()` and `measure_objects_sep()`.
- `filter_type='matched'` in `get_objects_sextractor()`, so that the detection
  threshold is in units of the filtered noise, as in `sep.extract()`.
- `get_objects_sep()` reports background flux in the aperture (`bg_flux`,
  `bg_fluxerr`), and per-source `fwhm_model` and `aper` columns with a
  position-dependent FWHM model. `measure_objects_sep()` also reports
  `bg_fluxerr`.
- VizieR mirror fallback in `get_cat_vizier()` (`catalogs.vizier_mirrors`,
  used on network errors only) and a `timeout` argument.
- Sesame mirror fallback in `resolve` (`resolve.sesame_mirrors`), and
  `resolve()` falls through to the next method when Simbad or TNS lookups
  fail.
- `cache` option in `templates.get_hips_image()`.
- `utils.file_lock()` and inter-process locking of skycell cache downloads in
  `templates.get_skycells()`.
- `plots.render_lock` serializing Matplotlib rendering in `figure_saver`
  across threads.
- Support for photutils 3.0.

### Fixed

- Local background in `measure_objects()` was truncated to integers.
- Background aperture noise in `get_objects_sep()` counted the sky Poisson
  noise twice.
- Masked object positions were measured at the values hidden under the mask.
- PSF fits of stars with masked pixels nearby were discarded as failed.
- PSF sub-pixel phase errors: centre of even-sized PSFEx models, conversion
  of sub-pixel integrated models in the photutils backend, and resampling of
  point-sampled models in the SEP backend.
- `get_psf_stamp()` lost >5% of the peak of even-sized point-sampled models.
- Robust `photometry_model.match()` with statsmodels 0.15, and huge cancelling
  coefficients for rank-deficient designs; errors of terms not constrained by
  the data are now NaN.
- Catalogue band augmentation: Gaia DR2 produced no Johnson-Cousins
  magnitudes since astroquery 0.4.8, Gaia EDR3 was treated as DR2, and
  `augment_bands=False` was ignored in `get_cat_vizier()`.
- Legacy Survey bricks present in both northern and southern lists were
  always taken from the south, and bricks lacking the requested band produced
  failed downloads.
- Remote FITS downloads no longer leave temporary files or fill the astropy
  download cache.
- Three-color images in `plots.imshow()` are now properly normalized.
- `refine_wcs_simple(method='astrometrynet')` passed unmatched objects to
  `fit-wcs`.

## 0.4.1 - 2026-08-07

### Fixed

- NumPy 2 incompatibilities in flag handling and set membership.
- A perfect photometric fit in `photometry_model.match()` is treated as
  success rather than a degenerate model.

## 0.4.0 - 2026-08-02

First release published through the automated PyPI release workflow.
