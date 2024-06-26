=============================================
STDPipe - Simple Transient Detection Pipeline
=============================================

*AKA: random codes noone else will ever use*

*STDPipe* is a set of Python routines for astrometry, photometry and transient detection related tasks, intended for quick and easy implementation of custom pipelines, as well as for interactive data analysis.

Design principles:
 - implemented as a library of routines covering most common tasks
 - operates on standard Python objects: NumPy arrays for images, Astropy Tables for catalogs and object lists, etc
 - does not try to re-implement the things already implemented in other Python packages
 - conveniently wraps external codes that do not have their own Python interfaces (*SExtractor*, *SCAMP*, *PSFEx*, *HOTPANTS*, *Astrometry.Net*, ...):

     - wrapping is transparent: all data passed from Python, all options customizable from Python, all (or most of) outputs available back
     - everything operates on temporary files, nothing is kept after the run unless explicitly asked for

Quick Start
-----------

See `notebooks/stdpipe_tutorial.ipynb <https://github.com/karpov-sv/stdpipe/blob/master/notebooks/stdpipe_tutorial.ipynb>`_ for an example notebook demonstrating basic steps of a typical image processing with *STDPipe*. This notebook uses the image available
`here <https://pc048b.fzu.cz/~karpov/20210222223821-052-RA.fits.processed.fits>`_.

User Guide
----------

.. toctree::
   :maxdepth: 3

   About <self>
   installation
   usage
   contributing
   todo
   API documentation <api/modules>

Contributing
------------

*STDPipe* is released under the MIT license.  We encourage you to
modify it, reuse it, and contribute changes back for the benefit of
others.  We follow standard open source development practices: changes
are submitted as pull requests and, once they pass the test suite,
reviewed by the team before inclusion.  Please also see
:doc:`our contributing guide <contributing>`.

Authors
-------

*STDPipe* is primarily written by Sergey Karpov.

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
