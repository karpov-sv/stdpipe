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
     - PanSTARRS DR1 or Legacy Survey templates
     - HOTPANTS + custom noise model
     - ZOGY
 - transient detection and photometry
     - noise-weighted detection, cutout adjustment, ...
 - auxiliary functions
     - PSF estimation, simulated stars, FITS header utilities, plotting, ...
 - light curve creation (soon)
     - spatial clustering, color regression, variability analysis, ...

# Installation

*STDpipe* is available at https://github.com/karpov-sv/stdpipe and is mirrored at https://gitlab.in2p3.fr/icare/stdpipe

The package is in constant development, so to keep track of the changes the suggested way of installing it is by cloning the repository
```
git clone https://github.com/karpov-sv/stdpipe.git
```
and then installing from it in development (or "editable") mode by running the command
```
cd stdpipe
python3 -m pip install -e .
```
This way you may update the repository or apply local patches, and it will immediately be reflected in the installed package.

Apart of Python requirements that will be installed automatically, *STDPipe* also (optionally) makes use of the following external software:
 - [SExtractor](https://github.com/astromatic/sextractor)
 - [SCAMP](https://github.com/astromatic/scamp)
 - [PSFEx](https://github.com/astromatic/psfex)
 - [SWarp](https://github.com/astromatic/swarp)
 - [HOTPANTS](https://github.com/acbecker/hotpants)
 - [Astrometry.Net](https://github.com/dstndstn/astrometry.net)

Most of them may be installed from your package manager. E.g. on Debian or Ubuntu systems it may look like that:
```
sudo apt install sextractor scamp psfex swarp
```
or, on Miniconda/Anaconda, like that:
```
conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-psfex astromatic-swarp
```

You may also check more detailed installation instructions [here](https://stdpipe.readthedocs.io/en/latest/installation.html).

# Usage

Please consult the [documentation for *STDPipe*](https://stdpipe.readthedocs.io/) for the basic usage patterns and description of its API. You may check the examples inside [notebooks/](notebooks/) folder, especially the [tutorial](notebooks/stdpipe_tutorial.ipynb) that demonstrates basic steps of a typical image processing.
