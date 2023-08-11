#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__version__ = '0.1'

requirements = [
    'numpy',
    'scipy',
    'astropy>=4.1',
    'matplotlib',
    'astroquery>0.4.1',
    'sep',
    'astroscrappy',
    'photutils',
    'statsmodels',
    'tqdm',
    'regions',
    'python-dateutil',
    'requests',
    'sip_tpv',
    'pyfftw',
]

setup(
    name='stdpipe',
    version=__version__,
    description='Simple Transient Detection Pipeline',
    author='Sergey Karpov',
    author_email='karpov.sv@gmail.com',
    url='https://github.com/karpov-sv/stdpipe',
    install_requires=requirements,
    packages=['stdpipe'],
    package_data={'stdpipe':['data/*']},
    include_package_data=True,
)
