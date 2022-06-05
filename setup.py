#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__version__ = '0.1'

requirements = [
    'configargparse',
    'numpy',
    'scipy',
    'astropy>=4.1',
    'matplotlib',
    'pandas',
    'astroquery>0.4.1',
    'sep',
    'photutils',
    'statsmodels',
    'tqdm',
    'regions',
    'dateutil',
    'requests',
]

setup(
    name='stdpipe',
    version=__version__,
    description='Simple Transient Detection Pipeline',
    author='Sergey Karpov',
    author_email='karpov.sv@gmail.com',
    url='',
    install_requires=requirements,
    packages=['stdpipe'],
    package_data={'stdpipe':['data/*']},
    include_package_data=True,
    extras_require={
        "example": [
            'astroscrappy',
        ],
    },
)
