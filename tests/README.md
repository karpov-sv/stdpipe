# STDPipe Tests

This directory contains the test suite for STDPipe.

## Quick Start

```bash
# Install test dependencies
pip install -r requirements_test.txt

# Run all tests
pytest

# Run only fast unit tests
pytest -m unit

# Run with coverage
pytest --cov=stdpipe --cov-report=html
```

## Test Structure

- `conftest.py` - Shared fixtures and pytest configuration
- `test_*.py` - Test modules for each STDPipe module
- `*.fits` - Test data files (real astronomical images)

## Test Categories

- **Unit tests** (`@pytest.mark.unit`) - Fast, no external dependencies
- **Integration tests** (`@pytest.mark.integration`) - Require external tools
- **Slow tests** (`@pytest.mark.slow`) - Time-consuming tests

## Available Test Modules

| Test File | Coverage |
|-----------|----------|
| `test_astrometry.py` | Coordinate transformations, WCS utilities |
| `test_catalogs.py` | Catalog queries and processing |
| `test_photometry.py` | Object detection, SEP, SExtractor |
| `test_pipeline.py` | Image masking and preprocessing |
| `test_utils.py` | Utility functions and helpers |

## Test Data

FITS files in this directory are used for integration testing:
- Various real astronomical images from different instruments
- Images with known sources for photometry testing
- Images for stacking and subtraction tests

## External Tool Requirements

Some integration tests require external software:
- SExtractor
- SCAMP
- PSFEx
- HOTPANTS
- Astrometry.Net

Tests requiring unavailable tools are automatically skipped.

## More Information

See [TESTING.md](../TESTING.md) for comprehensive testing documentation.
