#!/bin/sh

# Basic dependencies
conda install numpy scipy astropy matplotlib pandas

# conda-forge dependencies
conda install -c conda-forge astroscrappy

# astropy dependencies
conda install -c astropy astroquery

# External binaries
conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-psfex astromatic-swarp
