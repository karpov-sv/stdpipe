Using STDPipe
=============

*STDPipe* is a library of routines that operate on standard Python objects, especially the ones from Numpy (two-dimensional arrays for images) and Astropy (tables, WCS solutions, FITS files, ...). Therefore, in order to use it, you will need to import all necessary packages first. E.g. first lines of your script may look like that:

.. code-block:: python

   # Matplotlib is for plotting!
   import matplotlib.pyplot as plt
   # NumPy for data arrays
   import numpy as np

   # AstroPy WCS and FITS support
   from astropy.wcs import WCS
   from astropy.io import fits

   # Support for sky coordinates and times from AstroPy
   from astropy.coordinates import SkyCoord
   from astropy.time import Time

   # Disable some annoying warnings from AstroPy
   import warnings
   from astropy.wcs import FITSFixedWarning
   warnings.simplefilter(action='ignore', category=FITSFixedWarning)
   from astropy.utils.exceptions import AstropyUserWarning
   warnings.simplefilter(action='ignore', category=AstropyUserWarning)

Next, you will need to import the modules from *STDPipe* itself:

.. code-block:: python

   # Load (most of) our sub-modules
   from stdpipe import astrometry, photometry, catalogs, cutouts, templates, subtraction, plots, psf, pipeline, utils

Now you have everything imported, and may start actual data analysis!

Data processing
---------------

.. toctree::
   :maxdepth: 3

   preprocessing
   detection
   catalogs
   astrometry
   photometry
   subtraction
   transients
   cutouts

Common principles
-----------------

The functions included in *STDPipe* try to follow several common conventions related to their behaviour and arguments. They are summarized below.

Design principles:

- The routines operate on standard Python objects: NumPy arrays for images, Astropy Tables for catalogs and object lists, etc
- Some of them conveniently wrap external codes (*SExtractor*, *SCAMP*, *PSFEx*, *HOTPANTS*, *Astrometry.Net*, ...):

  - all data are passed from Python, all options are customizable from Python, all (or most of) outputs available back.
  - no configuration files are necessary or being used by default, but you may of course let the underlying executable to use a config file by passing corresponding options to it
  - everything operates on temporary files, nothing is kept after the run unless explicitly asked for
  - temporary files are created in unique temporary directories for each run, so several instances of routines may be safely run in parallel

Common conventions for routine arguments:

- Most of functions accept `verbose` argument that controls the amount of informational outputs the function produces. You may use `verbose=True` to see the details of what exactly the function is doing. Also, you may pass any `print`-like function to this argument to receive the messages instead of printing - so e.g. they may be logged.

.. code-block:: python

   # Define simple logging function
   def print_to_file(*args, logname='/tmp/logfile', clear=False, **kwargs):
       if clear and os.path.exists(logname):
           print('Clearing', logname)
           os.unlink(logname)

       if len(args) or len(kwargs):
           print(*args, **kwargs)
           with open(logname, 'a+') as lfd:
               print(file=lfd, *args, **kwargs)

   # verbose = True
   verbose = print_to_file

   # Now use it to redirect STDPipe function output to log file
   obj = photometry.get_objects_sextractor(image, mask=mask, verbose=verbose)

- Functions that produce (and then delete of course) some temporary files during its operation usually accept `_tmpdir` argument to manually specify the location where these temporary files (usually in a dedicated per-task temporary folder, so thread-safe and stuff) will be stored. This is useful if your system-wide temporary directory (usually :file:`/tmp` in Linux) is low on free space - then you may use some larger volume for storing temporary files by adding `_tmpdir=/large/volume/tmp` to function call.

- Functions that operate on temporary files in temporary folders may be supplied with `_workdir` argument - then they will store all temporary files related to their work in this specific folder, and will not remove them afterwards. So e.g. you will be able to directly see e.g what configuration files were created, manually re-run the failed command to experiment with its options (with `verbose=True` function call typically prints the complete command line of all calls to external programs, so you may just directly copy and paste it to terminal to repeat its invocation), etc.

- Functions that run external programs (e.g. SExtractor, HOTPANTS, or Astrometry.Net) usually accept `_exe` argument to directly specify the path to corresponding executable. If not specified, the code will try to automatically find it for you, so normally you do not need to worry about it.
