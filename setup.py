#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal setup.py for backward compatibility.

Modern configuration is in pyproject.toml.
This file is kept for backward compatibility with older pip versions
and tools that don't yet support PEP 517/518.

For development installation:
    pip install -e .

For installation with optional dependencies:
    pip install -e ".[dev]"
    pip install -e ".[docs]"
    pip install -e ".[all]"
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This setup.py is just for backward compatibility
if __name__ == "__main__":
    setup()
