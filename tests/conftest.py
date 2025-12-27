"""
Pytest configuration and fixtures for STDPipe tests.

This file contains shared fixtures and utilities used across all tests.
"""

import pytest
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


# ============================================================================
# Directory and Path Fixtures
# ============================================================================

@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Synthetic Image Fixtures
# ============================================================================

@pytest.fixture
def simple_image():
    """Create a simple 100x100 test image with Gaussian noise."""
    np.random.seed(42)
    image = np.random.normal(100, 10, (100, 100))
    return image.astype(np.float64)


@pytest.fixture
def image_with_sources():
    """Create a 256x256 image with artificial point sources on sky background."""
    np.random.seed(42)

    # Background with noise
    image = np.random.normal(100, 5, (256, 256))

    # Add some point sources (simple Gaussian profiles)
    sources = [
        (50, 50, 1000),   # x, y, amplitude
        (120, 80, 800),
        (200, 150, 1200),
        (80, 200, 600),
    ]

    y, x = np.ogrid[:256, :256]
    for sx, sy, amp in sources:
        # FWHM ~ 3 pixels, sigma ~ 1.27
        sigma = 1.27
        source = amp * np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * sigma**2))
        image += source

    return image.astype(np.float64)


@pytest.fixture
def simple_mask(simple_image):
    """Create a simple mask for the simple_image (no masked pixels)."""
    return np.zeros(simple_image.shape, dtype=bool)


@pytest.fixture
def mask_with_bad_pixels(image_with_sources):
    """Create a mask with some bad pixels marked."""
    mask = np.zeros(image_with_sources.shape, dtype=bool)
    # Mask some edge regions
    mask[:10, :] = True
    mask[-10:, :] = True
    mask[:, :10] = True
    mask[:, -10:] = True
    # Mask a small bad region
    mask[100:110, 100:110] = True
    return mask


# ============================================================================
# FITS Header and WCS Fixtures
# ============================================================================

@pytest.fixture
def simple_header():
    """Create a simple FITS header without WCS."""
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = -64
    header['NAXIS'] = 2
    header['NAXIS1'] = 100
    header['NAXIS2'] = 100
    header['EXPTIME'] = 60.0
    header['GAIN'] = 1.5
    header['RDNOISE'] = 5.0
    return header


@pytest.fixture
def header_with_wcs():
    """Create a FITS header with a simple WCS solution."""
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = -64
    header['NAXIS'] = 2
    header['NAXIS1'] = 256
    header['NAXIS2'] = 256
    header['EXPTIME'] = 60.0
    header['GAIN'] = 1.5

    # Add WCS keywords (centered on a random sky position)
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL1'] = 180.0  # RA in degrees
    header['CRVAL2'] = 45.0   # Dec in degrees
    header['CRPIX1'] = 128.5  # Reference pixel
    header['CRPIX2'] = 128.5
    header['CD1_1'] = -0.0002778  # ~1 arcsec/pixel
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = 0.0002778
    header['EQUINOX'] = 2000.0

    return header


@pytest.fixture
def simple_wcs(header_with_wcs):
    """Create a simple WCS object."""
    return WCS(header_with_wcs)


# ============================================================================
# Catalog and Table Fixtures
# ============================================================================

@pytest.fixture
def simple_catalog():
    """Create a simple catalog table with mock objects."""
    catalog = Table()
    catalog['ra'] = [180.0, 180.1, 180.2, 179.9]
    catalog['dec'] = [45.0, 45.1, 44.9, 45.05]
    catalog['mag'] = [15.0, 16.0, 14.5, 17.0]
    catalog['mag_err'] = [0.01, 0.02, 0.01, 0.05]
    catalog['flux'] = 10**((25 - catalog['mag']) / 2.5)
    return catalog


@pytest.fixture
def detected_objects():
    """Create a table mimicking SEP/SExtractor output."""
    objects = Table()
    objects['x'] = [50.0, 120.0, 200.0, 80.0]
    objects['y'] = [50.0, 80.0, 150.0, 200.0]
    objects['flux'] = [10000.0, 8000.0, 12000.0, 6000.0]
    objects['flux_err'] = [100.0, 120.0, 150.0, 200.0]
    objects['mag'] = 25 - 2.5 * np.log10(objects['flux'])
    objects['mag_err'] = 2.5 / np.log(10) * objects['flux_err'] / objects['flux']
    objects['flags'] = [0, 0, 0, 0]
    objects['fwhm'] = [3.0, 3.2, 2.8, 3.1]
    return objects


# ============================================================================
# Real FITS File Fixtures
# ============================================================================

@pytest.fixture
def sample_fits_file(test_data_dir):
    """Return path to a sample FITS file if it exists, otherwise skip test."""
    # Try to find any FITS file in test data
    fits_files = list(test_data_dir.glob("*.fits"))
    if not fits_files:
        pytest.skip("No FITS test data available")
    return str(fits_files[0])


@pytest.fixture
def sample_fits_data(sample_fits_file):
    """Load a sample FITS file and return image and header."""
    with fits.open(sample_fits_file) as hdul:
        # Get the first image extension
        for hdu in hdul:
            if hdu.data is not None and len(hdu.data.shape) == 2:
                return hdu.data.astype(np.float64), hdu.header
    pytest.skip("No suitable image extension found in FITS file")


# ============================================================================
# External Tool Availability Fixtures
# ============================================================================

@pytest.fixture
def has_sextractor():
    """Check if SExtractor is available."""
    return shutil.which('sex') is not None or shutil.which('sextractor') is not None


@pytest.fixture
def has_scamp():
    """Check if SCAMP is available."""
    return shutil.which('scamp') is not None


@pytest.fixture
def has_psfex():
    """Check if PSFEx is available."""
    return shutil.which('psfex') is not None


@pytest.fixture
def has_hotpants():
    """Check if HOTPANTS is available."""
    return shutil.which('hotpants') is not None


@pytest.fixture
def has_astrometry_net():
    """Check if Astrometry.Net is available."""
    return shutil.which('solve-field') is not None


@pytest.fixture
def has_pyraf():
    """Check if PyRAF is available."""
    try:
        import pyraf
        return True
    except ImportError:
        return False


# ============================================================================
# pytest configuration hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "requires_pyraf: mark test as requiring PyRAF/IRAF"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle skipping tests based on availability."""
    for item in items:
        # Auto-skip tests requiring external tools if not available
        if "requires_sextractor" in item.keywords:
            if not shutil.which('sex') and not shutil.which('sextractor'):
                item.add_marker(pytest.mark.skip(reason="SExtractor not available"))

        if "requires_scamp" in item.keywords:
            if not shutil.which('scamp'):
                item.add_marker(pytest.mark.skip(reason="SCAMP not available"))

        if "requires_psfex" in item.keywords:
            if not shutil.which('psfex'):
                item.add_marker(pytest.mark.skip(reason="PSFEx not available"))

        if "requires_hotpants" in item.keywords:
            if not shutil.which('hotpants'):
                item.add_marker(pytest.mark.skip(reason="HOTPANTS not available"))

        if "requires_astrometry_net" in item.keywords:
            if not shutil.which('solve-field'):
                item.add_marker(pytest.mark.skip(reason="Astrometry.Net not available"))

        if "requires_pyraf" in item.keywords:
            try:
                import importlib.util
                spec = importlib.util.find_spec("pyraf")
                if spec is None:
                    item.add_marker(pytest.mark.skip(reason="PyRAF not available"))
            except (ImportError, ValueError):
                item.add_marker(pytest.mark.skip(reason="PyRAF not available"))


# ============================================================================
# PyRAF Test Fixtures
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def initialize_pyraf_once():
    """
    Initialize PyRAF once at the start of the test session.

    This avoids the issue where IRAF initialization during test collection
    conflicts with pytest's stdin/stdout capturing.
    """
    try:
        # Check if pyraf is available
        import importlib.util
        spec = importlib.util.find_spec("pyraf")
        if spec is not None:
            # Import and initialize pyraf once, outside of captured output
            import sys
            from io import StringIO

            # Temporarily redirect stdout/stderr to avoid pytest capture conflicts
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            old_stdin = sys.stdin

            try:
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                # IRAF needs a real stdin, not pytest's capture
                import os
                sys.stdin = open(os.devnull, 'r')

                from pyraf import iraf
                # IRAF is now initialized

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                if sys.stdin != old_stdin:
                    sys.stdin.close()
                sys.stdin = old_stdin

    except Exception:
        # If initialization fails, tests will be skipped anyway
        pass
