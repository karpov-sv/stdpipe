"""
STDPipe - Simple Transient Detection Pipeline.

*STDPipe* is a set of Python routines for astrometry, photometry and transient
detection related tasks, intended for quick and easy implementation of custom
pipelines, as well as for interactive data analysis.
"""

__version__ = "0.3.0"

# Optional real-bogus classifier (requires TensorFlow)
try:
    from . import realbogus
except ImportError:
    # TensorFlow not installed, realbogus module not available
    realbogus = None
