# STDPipe - Simple Transient Detection Pipeline
*AKA: random codes noone else will ever use*

*STDPipe* is a set of Python routines for astrometry, photometry and transient detection related tasks, intended for quick and easy implementation of custom pipelines, as well as for interactive data analysis.

### Design principles
 - implemented as a library of routines covering most common tasks
 - operates on standard Python objects: NumPy arrays for images, Astropy Tables for catalogs and object lists, etc
 - does not try to re-implement the things already implemented in other Python packages
 - conveniently wraps external codes that do not have their own Python interfaces (*SExtractor*, *SCAMP*, *PSFEx*, *HOTPANTS*, *Astrometry.Net*, ...)
     - wrapping is transparent: all data passed from Python, all options customizable from Python, all (or most of) outputs available back
     - everything operates on temporary files, nothing is kept after the run unless explicitly asked for

### Features
 - ~~pre-processing~~ - should be handled before in an instrument-specific way
     - bias/dark subtraction, flatfielding, masking
 - object detection and photometry
     - SExtractor or SEP for detection, photutils for photometry 
 - astrometric calibration
     - Astrometry.Net for blind WCS solving
     - SCAMP or Astropy-based code for refinement
 - photometric calibration
     - Vizier catalogues, passband conversion (PS1 to Johnson, Gaia to Johnson, ...)
     - spatial polynomial + color term + intrinsic scatter
 - image subtraction
     - HiPS templates
     - PanSTARRS DR1 templates
     - hotpants + custom noise model
 - transient detection and photometry
     - noise-weighted detection, cutout adjustment, ...
 - auxiliary functions
     - PSF estimation, simulated stars, FITS header utilities, plotting, ...
 - light curve creation (soon)
     - spatial clustering, color regression, variability analysis, ...

# Installation

*STDpipe* is available at https://gitlab.in2p3.fr/icare/stdpipe and is mirrored at https://github.com/karpov-sv/stdpipe

You may either install it from GitHub directly as 
```
python3 -m pip install --user git+https://github.com/karpov-sv/stdpipe
```
or clone the repository and then execute
```
python3 setup.py develop --user
```

Apart of Python requirements that will be installed automatically, *STDPipe* also (optionally) makes use of the following external software:
 - [SExtractor](https://github.com/astromatic/sextractor)
 - [SCAMP](https://github.com/astromatic/scamp)
 - [PSFEx](https://github.com/astromatic/psfex)
 - [HOTPANTS](https://github.com/acbecker/hotpants)
 - [Astrometry.Net](https://github.com/dstndstn/astrometry.net)

# Usage

There is no documentation for *STDPipe* yet, but you may check the examples inside [notebooks/](notebooks/) folder, especially the [tutorial](notebooks/stdpipe_tutorial.ipynb) that demonstrates basic steps of a typical image processing.
