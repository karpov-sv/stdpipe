# Testing Guide for STDPipe

This document describes the testing infrastructure and best practices for STDPipe.

## Overview

STDPipe uses **pytest** as its testing framework. Tests are organized into:
- **Unit tests** - Fast tests of individual functions without external dependencies
- **Integration tests** - Tests that require external tools (SExtractor, HOTPANTS, etc.) or network access
- **Slow tests** - Tests that take significant time to complete

## Setup

### Install Test Dependencies

```bash
pip install -r requirements_test.txt
```

This installs:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution
- `pytest-mock` - Mocking utilities
- `pytest-timeout` - Timeout handling

### Optional Tools for Integration Tests

Some integration tests require external astronomical software:
- **SExtractor** - Object detection
- **SCAMP** - Astrometric calibration
- **PSFEx** - PSF modeling
- **HOTPANTS** - Image subtraction
- **Astrometry.Net** - Blind astrometry

Tests requiring these tools are automatically skipped if the tools are not available.

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_photometry.py
```

### Run Specific Test Function

```bash
pytest tests/test_photometry.py::TestSEPPhotometry::test_get_objects_sep_simple_image
```

### Run Only Unit Tests (Fast)

```bash
pytest -m unit
```

### Run Only Integration Tests

```bash
pytest -m integration
```

### Skip Slow Tests

```bash
pytest -m "not slow"
```

### Run Tests in Parallel

```bash
pytest -n auto  # Uses all available CPU cores
pytest -n 4     # Uses 4 cores
```

### Run with Coverage Report

```bash
pytest --cov=stdpipe --cov-report=html
```

Then open `htmlcov/index.html` in your browser.

### Run with Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_astrometry.py       # Tests for astrometry module
├── test_catalogs.py         # Tests for catalog queries
├── test_photometry.py       # Tests for photometry and detection
├── test_pipeline.py         # Tests for pipeline utilities
├── test_utils.py            # Tests for utility functions
├── *.fits                   # Test data files
└── ...
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Unit tests (no external dependencies)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.requires_sextractor` - Requires SExtractor
- `@pytest.mark.requires_scamp` - Requires SCAMP
- `@pytest.mark.requires_psfex` - Requires PSFEx
- `@pytest.mark.requires_hotpants` - Requires HOTPANTS
- `@pytest.mark.requires_astrometry_net` - Requires Astrometry.Net
- `@pytest.mark.requires_network` - Requires network access

### Example Usage

```python
@pytest.mark.unit
def test_simple_function():
    """Fast unit test."""
    assert 1 + 1 == 2

@pytest.mark.integration
@pytest.mark.requires_sextractor
def test_sextractor_wrapper():
    """Test that requires SExtractor to be installed."""
    # This will be skipped if SExtractor is not available
    pass
```

## Available Fixtures

The `tests/conftest.py` file provides many useful fixtures:

### Directory Fixtures

- `test_data_dir` - Path to test data directory
- `temp_dir` - Temporary directory (auto-cleaned)

### Image Fixtures

- `simple_image` - 100x100 noise image
- `image_with_sources` - 256x256 image with artificial point sources
- `simple_mask` - Empty mask
- `mask_with_bad_pixels` - Mask with edges and bad regions marked

### FITS Header Fixtures

- `simple_header` - Basic FITS header
- `header_with_wcs` - FITS header with WCS
- `simple_wcs` - WCS object

### Catalog Fixtures

- `simple_catalog` - Mock catalog table
- `detected_objects` - Mock detected objects table

### Real Data Fixtures

- `sample_fits_file` - Path to real FITS test file
- `sample_fits_data` - Loaded FITS image and header

### Tool Availability Fixtures

- `has_sextractor` - Boolean, True if SExtractor available
- `has_scamp` - Boolean, True if SCAMP available
- `has_psfex` - Boolean, True if PSFEx available
- `has_hotpants` - Boolean, True if HOTPANTS available
- `has_astrometry_net` - Boolean, True if Astrometry.Net available

## Writing Tests

### Unit Test Example

```python
import pytest
from stdpipe import photometry

@pytest.mark.unit
def test_make_kernel_basic():
    """Test basic kernel creation."""
    kernel = photometry.make_kernel(r0=1.0, ext=2.0)

    assert kernel.ndim == 2
    assert kernel.max() == 1.0
    assert kernel.sum() > 0
```

### Integration Test Example

```python
import pytest
from stdpipe import photometry

@pytest.mark.integration
@pytest.mark.requires_sextractor
def test_sextractor_detection(image_with_sources, temp_dir):
    """Test SExtractor object detection."""
    obj = photometry.get_objects_sextractor(
        image_with_sources,
        thresh=5.0,
        _workdir=temp_dir,
        verbose=False
    )

    assert len(obj) >= 4  # Should find our artificial sources
```

### Using Fixtures

```python
@pytest.mark.unit
def test_with_fixtures(simple_image, simple_mask, temp_dir):
    """Test using multiple fixtures."""
    # simple_image, simple_mask, and temp_dir are automatically provided
    assert simple_image.shape == simple_mask.shape

    # Use temp_dir for any temporary files
    import os
    output_file = os.path.join(temp_dir, 'output.fits')
    # ... save something ...
    # temp_dir will be automatically cleaned up
```

## Best Practices

### 1. Use Appropriate Markers

Always mark your tests appropriately:
```python
@pytest.mark.unit  # For fast, isolated tests
@pytest.mark.integration  # For tests requiring external tools
@pytest.mark.slow  # For time-consuming tests
```

### 2. Use Fixtures for Test Data

Don't create test data in every test - use fixtures:
```python
# Good
def test_something(simple_image):
    result = process(simple_image)
    assert result is not None

# Less good
def test_something():
    image = np.random.random((100, 100))  # Recreated every time
    result = process(image)
    assert result is not None
```

### 3. Test Both Success and Failure Cases

```python
def test_valid_input(simple_image):
    """Test with valid input."""
    result = process(simple_image)
    assert result is not None

def test_invalid_input():
    """Test error handling with invalid input."""
    with pytest.raises(ValueError):
        process(None)
```

### 4. Use Parametrize for Multiple Cases

```python
@pytest.mark.parametrize("threshold,expected_count", [
    (3.0, 10),
    (5.0, 5),
    (10.0, 1),
])
def test_detection_thresholds(image_with_sources, threshold, expected_count):
    """Test different detection thresholds."""
    obj = detect_objects(image_with_sources, thresh=threshold)
    assert len(obj) <= expected_count
```

### 5. Keep Tests Fast

- Use unit tests for most functionality
- Mock external dependencies when possible
- Reserve integration tests for critical workflows
- Mark slow tests appropriately

### 6. Test Edge Cases

```python
def test_empty_image():
    """Test with zero-sized image."""
    empty = np.array([])
    # Should handle gracefully

def test_single_pixel():
    """Test with 1x1 image."""
    single = np.array([[1.0]])
    # Should work or raise clear error
```

### 7. Clear Test Names and Docstrings

```python
# Good
@pytest.mark.unit
def test_spherical_distance_at_pole():
    """Test spherical distance calculation near the celestial pole."""
    # Clear what is being tested and why

# Less clear
def test_distance():
    """Test distance."""
    # What kind? Under what conditions?
```

## Continuous Integration

When setting up CI (GitHub Actions, etc.), consider:

```yaml
# Example GitHub Actions workflow
- name: Run unit tests
  run: pytest -m unit -v

- name: Run integration tests
  run: pytest -m integration -v

- name: Generate coverage
  run: pytest --cov=stdpipe --cov-report=xml
```

## Debugging Failed Tests

### Show Local Variables

```bash
pytest --showlocals
```

### Drop into Debugger on Failure

```bash
pytest --pdb
```

### Run Last Failed Tests

```bash
pytest --lf
```

### Run Only Tests That Failed

```bash
pytest --failed-first
```

### Increase Verbosity

```bash
pytest -vv --tb=long
```

## Coverage Goals

Aim for:
- **Core utilities** (astrometry, photometry): 80%+ coverage
- **Pipeline functions**: 70%+ coverage
- **Wrapper functions**: 60%+ coverage (harder to test without external tools)
- **Overall**: 70%+ coverage

Check coverage:
```bash
pytest --cov=stdpipe --cov-report=term-missing
```

## Adding New Tests

When adding new functionality:

1. **Write unit tests first** for core logic
2. **Add integration tests** if external tools are involved
3. **Use existing fixtures** when possible
4. **Mark tests appropriately** with decorators
5. **Update this guide** if adding new patterns or fixtures

## Example: Complete Test Module

```python
"""
Tests for new_module.py
"""

import pytest
import numpy as np
from stdpipe import new_module


class TestNewFunction:
    """Test the new_function utility."""

    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic operation."""
        result = new_module.new_function(42)
        assert result == 84

    @pytest.mark.unit
    def test_edge_case_zero(self):
        """Test with zero input."""
        result = new_module.new_function(0)
        assert result == 0

    @pytest.mark.unit
    def test_invalid_input(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            new_module.new_function(-1)


class TestNewIntegration:
    """Integration tests for new_module."""

    @pytest.mark.integration
    @pytest.mark.requires_sextractor
    def test_with_sextractor(self, image_with_sources, temp_dir):
        """Test integration with SExtractor."""
        result = new_module.process_with_sextractor(
            image_with_sources,
            _workdir=temp_dir
        )
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```
