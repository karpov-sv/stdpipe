# Instructions For NMMA Installation

## Preliminary Steps:

The steps highlighted below are primarily for Linux systems and Windows users are advised to use WSL (preferrably Ubuntu 20.04) for smooth installation. 
Ubuntu 20.04 is available for free download on Microsoft Store. 

**Installing Anaconda3**

On your Linux/WSL terminal, run the following commands to install anaconda (replace 5.3.1 by the latest version):


* $ wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh


* $ bash Anaconda3-5.3.1-Linux-x86_64.sh


(For 32-bit installation, skip the ‘_64’ in both commands)

NOTE: If you already have Anaconda3 installed, please make sure that it is updated to the latest version (conda update --all). Also check that you do not have multiple
versions of python installed in usr/lib/ directory as it can cause version conflicts while installing dependencies. 

Now do: 


* $ conda update --all


**Cloning the stdpipe repository**

Fork the stdpipe repository given below:


(stdpipe Github Repo)[https://github.com/karpov-sv/stdpipe]


Note that we link above to the main branch, but suggest making changes on your own fork (please also see our [contributing guide](./contributing.html)). Now, after forking, run the following command to clone the repository into your currently directory (by default, in your home directory):


* $ git clone https://github.com/your_github_username/stdpipe  
Change directory to the stdpipe folder:


* $ cd stdpipe


## Main Installation

Create a new environment using this command (environment name is stdpipe_env in this case):


* $ conda create --name stdpipe_env


* $ conda activate stdpipe_env


NOTE: If this gives an error like: CommandNotFoundError: Your shell has not been properly configured to use 'conda activate', then run:


* $ source ~/anaconda3/etc/profile.d/conda.sh


then proceed with conda activate nmma_env.

Check python and pip version like this:


* $ python --version
* $ pip --version


Python 3.7 and above and Pip 21.2 and above is ideal for this installation. It is recommended to update these for your installation. 


Install basic dependencies:


* $ conda install numpy scipy astropy matplotlib pandas

Install conda-forge dependencies

* $ conda install -c conda-forge esutil astroscrappy

Install conda astropy dependencies

* $ conda install -c astropy astroquery 

Use the commands below to install the dependencies given in requirements.txt file which are necessary for stdpipe: 


* $ python setup.py install

**First Test for stdpipe**

Run the following commands:

* $ ipython
* import stdpipe
* import stdpipe.photometry
* import stdpipe.cutouts

NOTE (Okay, last one!): if everything is ok, it's the end of the installation. But in case it shows that such-and-such modules are absent, feel free to install those modules by visiting their anaconda documentation and install
those with their given commands. In case modules like photutils and statsmodels are needed, don't hesitate to do it with pip (normally it shouldn't happen), but some modules may not install correctly in case of disturbance.

This instruction file will likely cover the issues you might face during your installation. However, please open issues on GitHub if there appear to be unresolvable conflicts. 

## Installation of useful (optional) packages

 - [SExtractor](https://github.com/astromatic/sextractor)
 - [SCAMP](https://github.com/astromatic/scamp)
 - [PSFEx](https://github.com/astromatic/psfex)
 - [HOTPANTS](https://github.com/acbecker/hotpants)
 - [Astrometry.Net](https://github.com/dstndstn/astrometry.net)

