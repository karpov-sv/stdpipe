# Building image from
FROM ubuntu:20.04

LABEL maintainer="Sergey Karpov <karpov.sv@gmail.com>"

# Create a variable corresponding to the name of the new user that will be created inside the Docker.
ENV USERNAME newuser

# Create new user
RUN useradd -ms /bin/bash ${USERNAME}

# Set the working directory
WORKDIR /home

# Fix for tzdata which is pulled by some dependencies
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Prague

# Install updates to base image, install packages, and remove from cache
# Install hotpants
# Install Jupyter
# Install some Python packages
# All done in one line to limit docker size
RUN  \
    apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates libtool libcfitsio-dev autoconf automake git unzip python3-pip \
    sextractor scamp psfex swarp \
    jupyter jupyter-notebook jupyter-nbextension-jupyter-js-widgets python3-ipywidgets \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/acbecker/hotpants.git \
    && cd hotpants \
    && make \
    && cp hotpants /usr/local/bin/ \
    && cd .. \
    && rm -fr hotpants \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install setuptools

# Install STDPipe
RUN  \
    git clone https://github.com/karpov-sv/stdpipe.git \
    && cd stdpipe \
    && python3 -m pip install -e . \
    && rm -fr /home/newuser/.cache/pip \
    && rm -fr /root/.cache/pip \
    && rm -fr /root/.cache/* \
    && rm -fr /tmp/* \
    && apt-get autoremove --purge -y

# switch ownership
# (all commands are root until a USER command changes that)
USER ${USERNAME}

# change ownership from root to USR:
# Create `stdpipe` directory to link on volume with host machine
RUN chown -R ${USERNAME}:${USERNAME}  /home/${USERNAME} \
    && mkdir /home/${USERNAME}/stdpipe/

# Add local bin to PATH
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

# Change working directory to stdpipe
WORKDIR /home/${USERNAME}/stdpipe

# define entrypoint which is the default executable at launch
ENTRYPOINT ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''"]

EXPOSE 8888
