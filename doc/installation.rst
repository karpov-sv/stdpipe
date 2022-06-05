Installation
============

*STDpipe* is available at https://github.com/karpov-sv/stdpipe and is mirrored at https://gitlab.in2p3.fr/icare/stdpipe.

In order to use it, you will need a working Python (>=3.6) installation, as well as a number of additional Python libraries and external packages. Below you will find some basic instructions how to set it up (using Anaconda environment as an example, but you may use any other as well) if you do not have it already.

Preparing the environment (optional)
------------------------------------

The steps highlighted below are primarily for Linux and MacOS systems.
Windows users are advised to use WSL (preferrably Ubuntu 20.04) for smooth installation.
Ubuntu 20.04 is available for free download on Microsoft Store.

You may safely skip these steps if you already have a working Python environment where you would like to install *STDPipe*.

Installing Anaconda
^^^^^^^^^^^^^^^^^^^

On your Linux/MacOS/WSL terminal, run the following commands to install `Anaconda <https://www.anaconda.com>`_ (replace 5.3.1 by the latest version, and adjust operating system name):

.. prompt:: bash

   wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh

.. prompt:: bash

   bash Anaconda3-5.3.1-Linux-x86_64.sh

(For 32-bit installation, skip the ‘_64’ in both commands)


NOTE: If you already have Anaconda3 installed, please make sure that it is updated to the latest version (:code:`conda update --all`). Also check that you do not have multiple
versions of python installed in usr/lib/ directory as it can cause version conflicts while installing dependencies.

Now do:

.. prompt:: bash

   conda update --all


Installing basic dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt:: bash

   conda install numpy scipy astropy matplotlib pandas

Install conda-forge dependencies

.. prompt:: bash

   conda install -c conda-forge astroscrappy

Install conda astropy dependencies

.. prompt:: bash

   conda install -c astropy astroquery


STDPipe installation
--------------------

Clone the STDPipe repository from GitHub at https://github.com/karpov-sv/stdpipe

.. prompt:: bash

   git clone https://github.com/karpov-sv/stdpipe.git

Change directory to the stdpipe folder:

.. prompt:: bash

   cd stdpipe

Use the command below to install the rest of dependencies and the package itself in an *editable* manner so that it will be updated automatically when you update the code:

.. prompt:: bash

   python setup.py develop

.. note::

   Alternative installation command (try it if the one above fails - they use slightly different strategies of installing the dependencies, so results may really vary!) would be

   .. prompt:: bash

      pip install -e .

Keeping up to date
^^^^^^^^^^^^^^^^^^

The command above installs the package to your Python environment in an *editable* way - it means that all changes you may make to the source tree (where you cloned the code) will immediately be reflected in the installed package, you do not need to repeat the installation.

As the code base in the repository evolves fast -- new features are being added, bugs fixed, etc -- it is a good idea to update your cloned code from the upstream often. The following command from inside stdpipe folder will do it:

.. prompt:: bash

   git pull


Quick testing the installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following commands inside your python (e.g. after typing :code:`ipython`):

.. prompt:: python

   import stdpipe
   import stdpipe.photometry
   import stdpipe.cutouts

NOTE (Okay, last one!): if everything is ok, it's the end of the installation. But in case it shows that such-and-such modules are absent, feel free to install those modules by visiting their anaconda documentation and install
those with their given commands. In case modules like photutils and statsmodels are needed, don't hesitate to do it with pip (normally it shouldn't happen), but some modules may not install correctly in case of disturbance.

This instruction page will likely cover the issues you might face during your installation. However, please open issues on GitHub if there appear to be unresolvable conflicts.

Installation of external packages
---------------------------------

*STDPipe* makes use of a number of (optional) external packages:

- `SExtractor <https://github.com/astromatic/sextractor>`__
- `SCAMP <https://github.com/astromatic/scamp>`__
- `PSFEx <https://github.com/astromatic/psfex>`__
- `SWarp <https://github.com/astromatic/swarp>`__
- `HOTPANTS <https://github.com/acbecker/hotpants>`__
- `Astrometry.Net <https://github.com/dstndstn/astrometry.net>`__

Most of them are also available in the repositories of various Linux distributions, and may be conveniently installed from there (see below).

HOTPANTS image subtraction package cannot presently (as far as I know) be installed from any package manager, and has to be compiled manually.

.. attention::

   If HOTPANTS compilation fails for you on the linking stage with a number of :code:`multiple definition of` error messages - that's a `known bug <https://github.com/acbecker/hotpants/issues/5>`__ related to some recent changes in GCC compiler defaults. You may easily fix it by editing the :file:`Makefile` and adding :code:`-fcommon` switch among the others in the `COPTS` options (line `30 <https://github.com/acbecker/hotpants/blob/master/Makefile#L30>`__ at the moment of writing).

Ubuntu
^^^^^^

.. prompt:: bash

   sudo apt install sextractor scamp psfex swarp

Astrometry.Net may also be installed from repository, but might require additional manual configuration steps (and quite a lot of disk space for larger indices!), so install it only when you really need it, and when you really know what you are doing!

.. prompt:: bash

   sudo apt install astrometry.net

Anaconda
^^^^^^^^

.. prompt:: bash

   conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-psfex astromatic-swarp
