# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**STDPipe** (Simple Transient Detection Pipeline) is a Python library for astronomical image processing, focusing on astrometry, photometry, and transient detection. It operates on standard Python objects (NumPy arrays for images, Astropy Tables for catalogs) and wraps external astronomical tools with transparent Python interfaces.

Key design principle: All external tool execution happens via temporary files - nothing is kept after the run unless explicitly requested.

## Installation and Development

### Modern Build System (pyproject.toml)

STDPipe uses modern Python packaging standards (PEP 517/621) with all configuration in `pyproject.toml`.

Install in editable/development mode:
```bash
# Basic editable install
python3 -m pip install -e .

# With development tools (pytest, coverage, etc.)
python3 -m pip install -e ".[dev]"

# With documentation tools
python3 -m pip install -e ".[docs]"

# Everything
python3 -m pip install -e ".[all]"
```

This allows immediate reflection of local changes without reinstalling. The package is in constant development, so editable mode is recommended.

### Building Distributions

Build source and wheel distributions:
```bash
python3 -m pip install build
python3 -m build
```

Output: `dist/stdpipe-{version}.tar.gz` and `dist/stdpipe-{version}-py3-none-any.whl`

### External Dependencies

STDPipe wraps these external astronomical tools (optional but commonly used):
- **SExtractor** - Object detection and photometry
- **SCAMP** - Astrometric calibration
- **PSFEx** - PSF model extraction
- **SWarp** - Image resampling
- **HOTPANTS** - Image subtraction
- **Astrometry.Net** - Blind WCS solving

Install via package manager:
```bash
# Debian/Ubuntu
sudo apt install sextractor scamp psfex swarp

# Conda
conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-psfex astromatic-swarp
```

HOTPANTS must be compiled manually - see `install_hotpants.sh` for installation script.

## Testing

Comprehensive pytest-based test suite with 783 tests covering core functionality (including 47 SFFT subtraction tests, 16 optimal extraction tests, 18 photutils PSF photometry tests, 17 DAOPHOT PSF photometry tests, and 32 photutils object detection tests).

### Running Tests

```bash
# Run all tests
pytest

# Run only fast unit tests
pytest -m unit

# Run with coverage report
pytest --cov=stdpipe --cov-report=html

# Run in parallel (faster)
pytest -n auto

# Run specific test file
pytest tests/test_photometry.py

# Run specific test function
pytest tests/test_photometry.py::TestSEPPhotometry::test_get_objects_sep_simple_image
```

### Test Categories

Tests are marked by category:
- `unit` - Fast tests without external dependencies (including 47 SFFT subtraction tests + 16 optimal extraction tests + 18 photutils PSF tests + 17 DAOPHOT PSF tests + 32 photutils detection tests)
- `integration` - Tests requiring external tools like SExtractor (6 tests)
- `slow` - Time-consuming tests
- `requires_sextractor`, `requires_hotpants`, `requires_pyraf`, etc. - Auto-skip if tool not installed
- `requires_network` - Tests needing internet access

### Test Configuration

All test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`. The test suite includes:
- 20+ reusable fixtures (images, WCS, catalogs, temp directories)
- Automatic skipping when external tools unavailable
- Property-based testing with hypothesis
- Coverage reporting integrated

See `TESTING.md` for comprehensive testing documentation.

## Code Architecture

### Module Organization

The codebase is organized into focused modules in `stdpipe/`:

**Core Processing Modules:**
- `pipeline.py` - Higher-level building blocks combining lower-level functionality. Contains utilities like `make_mask()` for image masking (cosmic rays, saturation, bad pixels) and other multi-step operations
- `photometry.py` - **Object detection and basic photometry**. Main functions:
  - `get_objects_sep()` - fast SEP-based detection (compiled Python). Supports position-dependent auto-FWHM via `fwhm_spatial_order`; stores the resulting `FWHMMap` in `obj.meta['fwhm_phot_model']` and applies it per source for aperture/bkgann/optimal PSF width.
  - `get_objects_sextractor()` - external SExtractor tool wrapper
  - `get_objects_photutils()` - pure Python photutils-based detection with 3 methods (segmentation with deblending, DAOStarFinder, IRAFStarFinder). **Key features**: optional deblending for crowded fields, multiple detection algorithms, extended API exposing photutils parameters. Drop-in replacement for `get_objects_sep()`.
  - `estimate_fwhm_from_objects()` - robust mode-based FWHM from a detection table. Scalar by default; `spatial_order >= 1` returns a `FWHMMap` callable (2-D polynomial `FWHM(x, y)`) with transparent scalar fallback on low candidate counts.
  - `FWHMMap` - callable polynomial FWHM(x, y) (scalar via `float(fmap)`), usable anywhere a `fwhm` argument is accepted.
- `photometry_measure.py` - **Aperture and optimal extraction photometry**. Main function: `measure_objects()` for photometry at detected object positions. Supports both standard aperture photometry and optimal extraction (Naylor 1998) that provides ~10% S/N improvement for point sources. **Advanced features**: grouped optimal extraction for crowded fields (simultaneously fits overlapping sources via weighted least squares). Handles background estimation, error propagation, centroiding, and local background annuli. Both `measure_objects` and `measure_objects_sep` accept a callable `fwhm` (e.g. `FWHMMap`): the SEP backend broadcasts per-source aperture/bkgann/fwhm arrays to SEP; the photutils backend uses per-source width in ungrouped optimal extraction and falls back to the scalar median elsewhere.
- `photometry_model.py` - **Photometric modeling and calibration**. Main functions: `match()` for photometric matching with spatial zero-point models, `make_sn_model()` for S/N modeling, `get_detection_limit_sn()` for detection limit estimation. Core calibration routines used by higher-level workflows.
- `photometry_psf.py` - **PSF photometry** using photutils. Main functions: `measure_objects_psf()` for PSF fitting photometry, `create_psf_model()` for building empirical ePSF. Provides more accurate flux measurements than aperture photometry, especially in crowded fields. Supports Gaussian, PSFEx, and empirical PSF models. **Advanced features**: position-dependent PSF (evaluates PSFEx polynomials at each position) and grouped PSF fitting (simultaneously fits overlapping sources).
- `photometry_iraf.py` - **IRAF DAOPHOT backend** for aperture and PSF photometry. Main functions: `measure_objects()` for aperture photometry using DAOPHOT phot task, `measure_objects_psf()` for PSF photometry using complete DAOPHOT workflow (phot → psf → allstar). Requires PyRAF/IRAF. Provides classic, battle-tested algorithms as alternative to photutils.
- `astrometry.py` - Astrometric calibration and coordinate transformations. Wraps Astrometry.Net for blind solving, SCAMP for refinement. Utilities for WCS, pixel scales, spherical distances
- `subtraction.py` - Image subtraction wrappers. Main functions: `run_sfft()` (built-in SFFT), `run_hotpants()` (external HOTPANTS), `run_zogy()` (built-in ZOGY). All share the same interface pattern (masks, error maps, gain, `get_noise`/`get_scaled`/`get_convolved`)
- `sfft.py` - SFFT (Hu et al. 2022) low-level solver. Main function: `solve()` for spatially varying kernel fitting in a single global least-squares problem. Helper functions: `evaluate_kernel_at()`, `evaluate_flux_scale()`, `evaluate_background()`
- `templates.py` - Template image acquisition from HiPS surveys (Pan-STARRS, Legacy Survey). Main function: `get_hips_image()` using CDS hips2fits service
- `catalogs.py` - Online catalog queries via Vizier. Supports PS1, Gaia DR2/EDR3/DR3, SDSS, ATLAS-REFCAT2, SkyMapper, etc. with automatic photometric augmentation (magnitude conversions). Main function: `get_cat_vizier()`
- `psf.py` - PSF modeling via PSFEx wrapper. Main function: `run_psfex()`

**Supporting Modules:**
- `cutouts.py` - Cutout extraction and manipulation
- `utils.py` - FITS I/O helpers, downloading, coordinate formatting, header parsing
- `plots.py` - Visualization utilities for images and catalogs
- `lcs.py` - Light curve handling
- `resolve.py` - Object name resolution
- `db.py` - Database interactions (likely for PostgreSQL/psycopg2)

### Data Flow Pattern

Typical processing pipeline:
1. **Load image** (NumPy array) + optional mask/header
2. **Preprocessing** (`pipeline.make_mask()`) - mask bad pixels, cosmic rays, saturation
3. **Object detection** - Choose one approach:
   - **SEP**: `photometry.get_objects_sep()` - fast, compiled, SExtractor-like
   - **photutils**: `photometry.get_objects_photutils()` - pure Python, 3 methods (segmentation/DAOStarFinder/IRAFStarFinder), optional deblending
   - **SExtractor**: `photometry.get_objects_sextractor()` - external tool, most features
   All return Astropy Table with identical column structure
4. **Astrometry** (`astrometry.py` functions) - solve/refine WCS
5. **Photometry** - Choose one approach:
   - **Aperture**: `photometry.measure_objects()` or `photometry_iraf.measure_objects()` - fast, robust, good for isolated sources
   - **PSF**: `photometry_psf.measure_objects_psf()` (photutils) or `photometry_iraf.measure_objects_psf()` (DAOPHOT) - more accurate, especially in crowded fields
6. **Photometric calibration** - match to catalogs (`catalogs.get_cat_vizier()`), fit zero-points
7. **Image subtraction** (optional) - get template (`templates.get_hips_image()`), subtract (`subtraction.run_sfft()`, `subtraction.run_hotpants()`, or `subtraction.run_zogy()`)
8. **Transient detection** - analyze difference image

### External Tool Wrapper Pattern

Modules wrapping external tools (SExtractor, SCAMP, PSFEx, HOTPANTS) follow a consistent pattern:

1. Accept NumPy arrays and Python parameters
2. Create temporary directory (cleaned up after, unless `_workdir` specified for debugging)
3. Write FITS files and configuration files to temp directory
4. Execute external binary with `subprocess`
5. Parse output files back into Python objects (tables, arrays)
6. Return results + optionally write to user-specified files

Key parameters for debugging wrappers:
- `_workdir` - Keep temporary files for inspection
- `_tmpdir` - Custom temporary directory location
- `_exe` - Override executable path
- `verbose` - Enable detailed logging

### Common Parameter Conventions

- `image` - NumPy array (2D, float)
- `mask` - Boolean array where `True` = masked/excluded pixels
- `header` - FITS header (astropy.io.fits.Header)
- `wcs` - Astrometric solution (astropy.wcs.WCS)
- `err` - Error/noise map (NumPy array). Can be `True` to auto-generate from image statistics
- `gain` - Detector gain in e-/ADU
- `verbose` - Boolean or callable (print-like function) for logging
- `extra` - Dictionary of additional parameters for external tools

### Photometry: Aperture vs Optimal vs PSF

**SEP-Based Photometry** (`photometry_measure.measure_objects_sep()`):
- Alternative backend using new SEP 1.4+ features
- Sigma-clipped local background via `sep.stats_circann()` (more robust in crowded fields)
- Grouped optimal extraction via `sep.sum_circle_optimal()`
- PSF fitting photometry via `sep.psf_fit()` with PSFEx or Gaussian models
- **Grouped flux-only fitting** (`group_sources=True, fit_positions=False`): NNLS deblending at fixed positions — fast, accurate for crowded fields with known positions
- **Fitting radius control** (`fit_radius`): limits pixels used in PSF fitting to a circular region (e.g., 2-3× FWHM), reducing contamination from distant neighbors
- C-based implementation (potentially faster)
- No polynomial background gradient fitting
- Excellent agreement with `measure_objects()` (<0.1% bias)
- Best for: crowded fields, performance-critical applications, consistency with SEP detection

**Aperture Photometry** (`photometry_measure.measure_objects()`):
- Sums flux within circular aperture
- Fast and robust
- Best for: isolated sources, quick measurements
- Truncates PSF wings at aperture edge

**Optimal Extraction** (`photometry_measure.measure_objects(optimal=True)`):
- Weighted sum using PSF profile (Naylor 1998 algorithm)
- ~10% S/N improvement over aperture for point sources
- Requires PSF model (`psf` parameter) or FWHM (`fwhm` parameter)
- **Uses pixel-integrated Gaussian PSF** - eliminates FWHM-dependent systematic bias:
  - Old: +11.6% at FWHM=1.5, +2.7% at FWHM=3.0
  - New: < 0.1% bias for all FWHM values
- **Grouped extraction** (enabled by default, `group_sources=True`):
  - Simultaneously fits overlapping sources via weighted least squares
  - Properly accounts for flux sharing between nearby sources
  - Dramatic accuracy improvement: 51× at 0.5 FWHM, 14× at 1.0 FWHM, 3× at 1.5 FWHM
  - No downside for isolated fields (identical results at >3 FWHM)
  - Returns `group_id` and `group_size` columns
  - Set `group_sources=False` only for sparse fields with performance constraints
- Best for: point sources, all crowding levels, when PSF is known

**PSF Photometry** (two backends available):

*Photutils Backend* (`photometry_psf.measure_objects_psf()`):
- Modern Python-based PSF fitting using photutils
- PSF models: Gaussian (auto), PSFEx, empirical ePSF
- Advanced features: position-dependent PSF, grouped fitting
- Fast, flexible, pure Python

*DAOPHOT Backend* (`photometry_iraf.measure_objects_psf()`):
- Classic IRAF DAOPHOT workflow (phot → psf → allstar)
- PSF models: Analytical functions (auto, gauss, moffat, lorentz, penny)
- Battle-tested, conservative fitting, widely cited
- Requires PyRAF/IRAF

Both approaches:
- More accurate than aperture photometry, especially in crowded fields
- Account for full PSF including wings
- Best for: accurate fluxes, crowded fields, faint sources
- Returns fitted positions (`x_psf`, `y_psf`) and quality metrics

**PSF Output Columns (all with `_psf` suffix):**
- `x_psf`, `y_psf` - Fitted source positions
- `qfit_psf` - Fit quality (0 = good)
- `cfit_psf` - Central pixel fit quality (0 = good)
- `flags_psf` - Photutils fit flags
- `npix_psf` - Number of unmasked pixels used in fit
- `reduced_chi2_psf` - Reduced chi-squared (photutils ≥ 2.3.0)

**Advanced PSF Features:**
- **Position-dependent PSF** (`use_position_dependent_psf=True`) - For wide-field imaging where PSF varies across the field. Evaluates PSFEx polynomial at each source position (constant, linear, quadratic terms). Slower but more accurate for fields with PSF variation.
- **Grouped PSF fitting** (enabled by default, `group_sources=True`) - Fits nearby sources simultaneously to reduce neighbor contamination. Dramatically more accurate in crowded fields (51× improvement at 0.5 FWHM, 14× at 1.0 FWHM, 3× at 1.5 FWHM). No downside for isolated fields (identical results at >3 FWHM). Set `group_sources=False` only for sparse fields with performance constraints. Configure with `grouper_radius` (default: 2× PSF size).
- **Combined mode** - Both features can be used together for maximum accuracy in crowded, wide-field images.

**Optimal Extraction Output Columns:**
- `flux`, `fluxerr` - Measured flux and error
- `npix_optimal` - Number of pixels used in extraction
- `chi2_optimal` - Reduced chi-squared of the fit
- `norm_optimal` - PSF normalization factor
- `group_id` - Group identifier (when `group_sources=True`)
- `group_size` - Number of sources in group (when `group_sources=True`)

**Optimal Extraction Flags (`flags` column):**
- `0x800` - Optimal extraction failed (NaN result)

**PSF Photometry Flags (`flags` column, in addition to standard flags):**
- `0x1000` - PSF fit failed (NaN result)
- `0x2000` - Large centroid shift during fit (>1 pixel)

**Examples:**
```python
from stdpipe import photometry, photometry_psf, psf
from stdpipe.photometry_measure import measure_objects_sep, _HAS_SEP_OPTIMAL

# Check if SEP optimal extraction is available
if _HAS_SEP_OPTIMAL:
    print("SEP 1.4+ features available")

# Object detection - SEP backend (fast, compiled)
obj = photometry.get_objects_sep(image, thresh=3.0, aper=5.0)

# Object detection - photutils backend (pure Python, 3 methods)
obj = photometry.get_objects_photutils(
    image,
    thresh=3.0,
    method='segmentation',  # or 'dao' or 'iraf'
    deblend=True,           # Deblend crowded sources
    aper=5.0
)

# photutils: Segmentation with deblending (crowded fields)
obj = photometry.get_objects_photutils(
    image,
    thresh=2.0,
    method='segmentation',
    deblend=True,
    nlevels=32,
    contrast=0.001,
    aper=3.0,
    sn=5.0
)

# photutils: DAOStarFinder (stellar fields)
obj = photometry.get_objects_photutils(
    image,
    thresh=5.0,
    method='dao',
    fwhm=3.0,
    aper=5.0
)

# Aperture photometry (fast)
from stdpipe import photometry_measure
result_aper = photometry_measure.measure_objects(obj, image, aper=3.0, fwhm=3.0)

# Optimal extraction (~10% S/N improvement for point sources)
# Grouped fitting is now enabled by default for safety
result_opt = photometry_measure.measure_objects(
    obj, image,
    aper=5.0,
    fwhm=3.0,
    optimal=True  # Uses Naylor 1998 algorithm, group_sources=True by default
)
# Check grouping results (for crowded fields)
print(result_opt['group_id'], result_opt['group_size'])

# For sparse fields with performance constraints (optional)
result_sparse = photometry_measure.measure_objects(
    obj, image,
    aper=5.0,
    fwhm=3.0,
    optimal=True,
    group_sources=False  # Disable grouped fitting (not recommended unless field is sparse)
)

# PSF photometry - Photutils backend (modern, flexible)
# Grouped fitting is now enabled by default for safety
result_psf = photometry_psf.measure_objects_psf(obj, image, fwhm=3.0)

# PSF photometry - DAOPHOT backend (classic, battle-tested)
from stdpipe import photometry_iraf
result_psf = photometry_iraf.measure_objects_psf(obj, image, fwhm=3.0)

# Photutils: Position-dependent PSF (wide-field images)
psfex_model = psf.run_psfex(image)
result = photometry_psf.measure_objects_psf(
    obj, image,
    psf=psfex_model,
    use_position_dependent_psf=True
)

# Photutils: Custom grouper radius (crowded fields)
result = photometry_psf.measure_objects_psf(
    obj, image,
    fwhm=3.0,
    grouper_radius=10.0  # Custom radius (group_sources=True by default)
)

# DAOPHOT: Custom PSF function and radii
result = photometry_iraf.measure_objects_psf(
    obj, image,
    fwhm=3.0,
    psf_function='moffat25',  # Force Moffat β=2.5
    psfrad=12.0,              # PSF radius
    fitrad=4.0,               # Fitting radius
    sn=5.0                    # S/N filter
)

# Photutils: Combined position-dependent + grouped (maximum accuracy)
result = photometry_psf.measure_objects_psf(
    obj, image,
    psf=psfex_model,
    use_position_dependent_psf=True,
    group_sources=True,
    grouper_radius=15.0
)

# Access quality metrics
good_fits = result[result['qfit_psf'] < 0.5]  # Low qfit = good quality
converged = result[result['cfit_psf'] < 0.01]  # Close to 0 = good quality in central pixel
if 'reduced_chi2_psf' in result.colnames:
    excellent = result[result['reduced_chi2_psf'] < 1.5]

# SEP-based photometry (requires SEP 1.4+ with new features)
if _HAS_SEP_OPTIMAL:
    # Aperture photometry with sigma-clipped local background
    result = measure_objects_sep(
        obj, image,
        aper=1.5, fwhm=3.0,
        bkgann=(3.0, 5.0),  # Sigma-clipped annulus background (robust)
        sn=5, verbose=True
    )

    # Grouped optimal extraction (crowded fields)
    result = measure_objects_sep(
        obj, image,
        aper=1.5, fwhm=3.0,
        optimal=True,
        group_sources=True,  # Automatic grouping in SEP
        sn=5, verbose=True
    )

    # PSF fitting with grouped deblending at fixed positions
    result = measure_objects_sep(
        obj, image,
        psf=psfex_model, fwhm=3.0,
        group_sources=True,
        fit_positions=False,     # Fix positions, only fit fluxes (NNLS)
        fit_radius=2.0 * 3.0,   # Limit fitting to 2× FWHM radius
        sn=5, verbose=True
    )
```

### Photometry Detection Flags

**Important:** SEP and SExtractor use different flag values for object detection quality. When using `get_objects_sep()`, consult https://sep.readthedocs.io/en/v1.1.x/reference.html for flag meanings (they differ from SExtractor).

## Documentation

Documentation is built with Sphinx and hosted on ReadTheDocs: https://stdpipe.readthedocs.io/

Build locally:
```bash
cd doc
make html
```

Tutorial notebook: `notebooks/stdpipe_tutorial.ipynb` demonstrates a complete processing workflow.

## Code Style Notes

- Uses `from __future__ import` for Python 2/3 compatibility (legacy)
- Extensive use of temporary file operations via `tempfile` module
- Functions include detailed docstrings with parameter descriptions
- Verbose logging pattern: `log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None`

## Behaviour patterns

- When debugging a regression or bug, always investigate the root cause first. Never 'fix' by reverting defaults or papering over symptoms without understanding why the issue occurs.
- After modifying any function signature or return type, immediately check and update all callers and all tests that reference that function. Run the full test suite before considering the task complete.
- When the user asks for an investigation or analysis, do the analysis first — do NOT jump to implementing code changes. Ask clarifying questions if the scope is unclear. Especially distinguish between 'evaluate/compare approaches' vs 'implement a fix'.
- Keep changes minimal and focused. Do not create summary documents, elaborate writeups, or refactor code unless explicitly asked. When reviewing or editing docstrings, edit them in-place — don't create separate files.
- When implementing features, prefer existing library conventions (e.g., use existing flags like `_HAS_SEP_OPTIMAL`, existing parameter naming patterns). Check existing code patterns before introducing new ones.
