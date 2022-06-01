ARG SCAMP_version=v2.6.7
ARG SWARP_version=2.38.0
ARG Sextractor_version=2.25.0
ARG PSFEX_version=3.21.1

# Building image from
FROM ubuntu:20.04

LABEL maintainer="Michael Coughlin <cough052@umn.edu>"

# Create a variable corresponding to the name of the new user that will be created inside the Docker.
ENV USR newuser

# Set up working directory
WORKDIR /home

# Create new user
RUN useradd -ms /bin/bash ${USR}

ARG SCAMP_version
ARG SWARP_version
ARG Sextractor_version
ARG PSFEX_version
# Install updates to base image, install packages, and remove from cache
# Create astromatic folder, and install all needed astromatic software and hotpants
# First download and build cfitsio and the cdsclient used by SCAMP
# Install all python packages. numpy 1.19.0 is the latest dependency compatible with tensorflow 2.3.1
# All done in one line to limit docker size
RUN  \
   apt-get update \
   && apt-get install software-properties-common -y \ 
   && add-apt-repository ppa:deadsnakes/ppa -y \ 
   && apt-get update \
   && apt-get install -y --no-install-recommends wget curl vim build-essential ca-certificates libtool libatlas3-base libatlas-base-dev libplplot-dev libfftw3-dev libcurl4-openssl-dev autoconf automake git unzip gfortran python3.9 python3.9-dev python3-pip \
   && apt-get install -y --no-install-recommends libcfitsio-dev sextractor scamp psfex swarp \
   && rm -rf /var/lib/apt/lists/* \
   && git clone https://github.com/acbecker/hotpants.git \
   && cd hotpants \
   && make \
   && cp hotpants /usr/local/bin/ \
   && cd .. \
   && rm -fr astromatic hotpants cfitsio-* cdsclient-* \
   && python3.9 -m pip install --upgrade pip \
   && python3.9 -m pip install setuptools cmake \
   && python3.9 -m pip install scikit-build  \
   && python3.9 -m pip install numpy scipy matplotlib astropy pandas shapely requests h5py scikit-image lacosmic hjson voevent-parse xmltodict astroML photutils keras keras-vis cython regions  opencv-python-headless astroscrappy astroquery tqdm 
RUN  \
   git clone https://github.com/karpov-sv/stdpipe.git \
   && cd stdpipe \
   && python3.9 setup.py install \
   && rm -fr /home/newuser/.cache/pip \
   && rm -fr /root/.cache/pip \
   && rm -fr /root/.cache/* \
   && rm -fr /tmp/* \
   && apt-get autoremove --purge -y
   #&& apt-get purge -y gfortran gfortran-7 git-man libgfortran-7-dev libgfortran4 


# switch ownership
# (all commands are root until a USER command changes that)
USER ${USR}

# Set the working directory
WORKDIR /home/${USR}

# change ownership from root to USR:
# Create directory to link on volume with host machine
RUN chown -R ${USR}:${USR}  /home/${USR} \
   && mkdir /home/${USR}/stdpipe/

# Add local bin to PATH
ENV PATH="/home/${USR}/.local/bin:$PATH"

# Change working directory to stdpipe
WORKDIR /home/${USR}/stdpipe

# define entrypoint which is the default executable at launch
ENTRYPOINT ["bash"]
