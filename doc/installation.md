# Installation


*STDpipe* is available at [https://github.com/karpov-sv/stdpipe](https://github.com/karpov-sv/stdpipe) and is mirrored at [https://gitlab.in2p3.fr/icare/stdpipe](https://gitlab.in2p3.fr/icare/stdpipe)

In order to use it, you will need a working Python (>=3.6) installation, as well as a number of additional Python libraries and external packages. Below you will find some basic instructions how to set it up (using Anaconda environment as an example, but you may use any other as well) if you do not have it already.


## Preparing the environment (optional)

The steps highlighted below are primarily for Linux and MacOS systems.
Windows users are advised to use WSL (preferrably Ubuntu 20.04) for smooth installation.
Ubuntu 20.04 is available for free download on Microsoft Store.

You may safely skip these steps if you already have a working Python environment where you would like to install *STDPipe*.


**Installing Anaconda**

On your Linux/MacOS/WSL terminal, run the following commands to install [Anaconda](https://www.anaconda.com) (replace 5.3.1 by the latest version, and adjust operating system name):

* $ wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh

* $ bash Anaconda3-5.3.1-Linux-x86_64.sh

(For 32-bit installation, skip the ‘_64’ in both commands)


NOTE: If you already have Anaconda3 installed, please make sure that it is updated to the latest version (conda update --all). Also check that you do not have multiple
versions of python installed in usr/lib/ directory as it can cause version conflicts while installing dependencies.

Now do:

* $ conda update --all


**Creating separate environment**

Create a new environment using this command (environment name is `stdpipe` in this case):

* $ conda create --name stdpipe
* $ conda activate stdpipe

NOTE: If this gives an error like:
    CommandNotFoundError:
Your shell has not been properly configured to use 'conda activate', then run:

* $ source ~/anaconda3/etc/profile.d/conda.sh

then proceed with conda activate nmma_env.

Check python and pip version like this:

* $ python --version
* $ pip --version

Python 3.7 and above and Pip 21.2 and above is ideal for this installation. It is recommended to update these for your installation.


**Installing basic dependencies**

* $ conda install numpy scipy astropy matplotlib pandas

Install conda-forge dependencies

* $ conda install -c conda-forge esutil astroscrappy

Install conda astropy dependencies

* $ conda install -c astropy astroquery


## STDPipe installation


Clone the STDPipe repository from GitHub at [https://github.com/karpov-sv/stdpipe](https://github.com/karpov-sv/stdpipe)

* $ git clone https://github.com/karpov-sv/stdpipe.git

Change directory to the stdpipe folder:

* $ cd stdpipe

Use the commands below to install the rest of dependencies and the package itself in an *editable* manner so that it will be updated automatically when you update the code:

* $ python setup.py develop


**Keeping up to date**

The command above installs the package to your Python environment in an *editable* way - it means that all changes you may make to the source tree (where you cloned the code) will immediately be reflected in the installed package, you do not need to repeat the installation.

As the code base in the repository evolves fast -- new features are being added, bugs fixed, etc -- it is a good idea to update your cloned code from the upstream often. The following command from inside stdpipe folder will do it:

* $ git pull


**Quick testing the installation**

Run the following commands:

* $ ipython
* import stdpipe
* import stdpipe.photometry
* import stdpipe.cutouts

NOTE (Okay, last one!): if everything is ok, it's the end of the installation. But in case it shows that such-and-such modules are absent, feel free to install those modules by visiting their anaconda documentation and install
those with their given commands. In case modules like photutils and statsmodels are needed, don't hesitate to do it with pip (normally it shouldn't happen), but some modules may not install correctly in case of disturbance.

This instruction file will likely cover the issues you might face during your installation. However, please open issues on GitHub if there appear to be unresolvable conflicts.

## Installation of external packages

*STDPipe* makes use of a number of (optional) external packages:

 - [SExtractor](https://github.com/astromatic/sextractor)
 - [SCAMP](https://github.com/astromatic/scamp)
 - [PSFEx](https://github.com/astromatic/psfex)
 - [SWarp](https://github.com/astromatic/swarp)
 - [HOTPANTS](https://github.com/acbecker/hotpants)
 - [Astrometry.Net](https://github.com/dstndstn/astrometry.net)

Most of them are also available in the repositories of various Linux distributions, and may be conveniently installed from there (see below).

HOTPANTS image subtraction package cannot presently (as far as I know) be installed from any package manager, and has to be compiled manually.

### Ubuntu

* $ sudo apt install sextractor scamp psfex swarp

Astrometry.Net may also be installed from repository, but might require additional manual configuration steps (and quite a lot of disk space for larger indices!), so install it only when you really need it, and when you really know what you are doing!

* $ sudo apt install astrometry.net

### Anaconda

* $ conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-psfex astromatic-swarp
