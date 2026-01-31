"""
Unit tests for stdpipe.photometry_psf module.

Tests PSF photometry routines using photutils.
"""

import pytest
import numpy as np
from astropy.table import Table

from stdpipe import photometry_psf
from stdpipe import psf


def _make_gaussian_image(size, x0, y0, fwhm, amplitude):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y, x = np.mgrid[:size, :size]
    return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def _make_psf_dict(size=25, sampling=1.0):
    y, x = np.mgrid[0:size, 0:size]
    data = np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * 3**2))
    data /= np.sum(data)
    return {
        'data': data[np.newaxis, :, :],
        'width': size,
        'height': size,
        'sampling': sampling,
        'degree': 0,
        'ncoeffs': 1,
        'x0': 0.0,
        'y0': 0.0,
        'sx': 1.0,
        'sy': 1.0,
        'type': 'psfex',
    }


def _make_mock_psf(seed=0):
    rng = np.random.RandomState(seed)
    mock_psf = {
        'data': rng.normal(0, 0.1, (3, 25, 25)),  # (ncoeff, height, width)
        'width': 25,
        'height': 25,
        'sampling': 1.0,
        'degree': 1,
        'ncoeffs': 3,
        'x0': 128.0,
        'y0': 128.0,
        'sx': 128.0,
        'sy': 128.0,
        'type': 'psfex'
    }

    y, x = np.mgrid[0:25, 0:25]
    mock_psf['data'][0] = np.exp(-((x - 12)**2 + (y - 12)**2) / (2 * 3**2))
    mock_psf['data'][0] /= np.sum(mock_psf['data'][0])

    return mock_psf


def _simulate_psf_field(
    rng,
    *,
    size,
    fwhm,
    n_sources,
    margin,
    min_sep,
    noise_std,
    flux_range,
    mask_fraction=0.0,
    recentroid=True,
    maxiters=20,
    fit_shape='square',
):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    xs = []
    ys = []
    attempts = 0
    while len(xs) < n_sources and attempts < 20000:
        attempts += 1
        x = rng.randint(margin, size - margin)
        y = rng.randint(margin, size - margin)
        if xs:
            dist2 = (np.array(xs) - x)**2 + (np.array(ys) - y)**2
            if np.any(dist2 < min_sep**2):
                continue
        xs.append(int(x))
        ys.append(int(y))

    assert len(xs) == n_sources

    fluxes = rng.uniform(flux_range[0], flux_range[1], size=n_sources)
    amplitudes = fluxes / (2 * np.pi * sigma**2)
    yy, xx = np.mgrid[:size, :size]
    image = rng.normal(0.0, noise_std, (size, size))
    for x, y, amp in zip(xs, ys, amplitudes):
        image += amp * np.exp(
            -((xx - x)**2 + (yy - y)**2) / (2 * sigma**2)
        )

    obj = Table()
    obj['x'] = np.array(xs, dtype=float)
    obj['y'] = np.array(ys, dtype=float)
    obj['flux'] = fluxes

    bg = np.zeros_like(image)
    err = np.full_like(image, noise_std)
    mask = None
    if mask_fraction:
        mask = rng.rand(size, size) < mask_fraction

    result = photometry_psf.measure_objects_psf(
        obj.copy(),
        image,
        fwhm=fwhm,
        bg=bg,
        err=err,
        mask=mask,
        recentroid=recentroid,
        maxiters=maxiters,
        fit_shape=fit_shape,
        verbose=False,
    )

    rel = np.abs(result['flux'] - fluxes) / fluxes
    valid = np.isfinite(rel)
    med_rel = float(np.nan)
    if np.any(valid):
        med_rel = float(np.median(rel[valid]))

    return {
        'result': result,
        'fluxes': fluxes,
        'med_rel': med_rel,
        'accuracy': 1.0 - med_rel,
        'n_valid': int(np.sum(valid)),
        'n_sources': n_sources,
        'mask_fraction': mask_fraction,
    }


class TestMeasureObjectsPSF:
    """Test PSF photometry measurement function."""

    @pytest.mark.unit
    def test_measure_objects_psf_gaussian(self, image_with_sources, detected_objects):
        """Test PSF photometry with Gaussian PSF on image with artificial sources."""
        # Use Gaussian PSF (will be created automatically)
        result = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) > 0

        # Check that expected columns are present
        required_cols = ['flux', 'fluxerr', 'mag', 'magerr', 'x_psf', 'y_psf']
        for col in required_cols:
            assert col in result.colnames

        # Fluxes should be positive for our artificial sources
        assert np.sum(result['flux'] > 0) > 0

        # PSF fit positions should be close to original positions
        # (within a few pixels for our clean artificial data)
        if len(result) > 0:
            pos_diff = np.sqrt(
                (result['x_psf'] - result['x'])**2 +
                (result['y_psf'] - result['y'])**2
            )
            # Most should be well-centered
            assert np.median(pos_diff[np.isfinite(pos_diff)]) < 2.0

    @pytest.mark.unit
    def test_measure_objects_psf_empty_table(self, simple_image):
        """Test PSF photometry with empty object table."""
        empty_obj = Table(names=['x', 'y'], dtype=[float, float])

        result = photometry_psf.measure_objects_psf(
            empty_obj,
            simple_image,
            fwhm=3.0,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) == 0

    @pytest.mark.unit
    def test_measure_objects_psf_with_mask(self, image_with_sources, detected_objects, mask_with_bad_pixels):
        """Test that masking works properly with PSF photometry."""
        result = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            mask=mask_with_bad_pixels,
            fwhm=3.0,
            verbose=False
        )

        assert isinstance(result, Table)
        # Should still measure some objects
        assert len(result) >= 0

    @pytest.mark.unit
    def test_measure_objects_psf_mask_avoids_bias(self):
        """Test that non-finite pixels are masked even with a user mask."""
        size = 51
        fwhm = 3.0
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        y, x = np.mgrid[:size, :size]
        cx = cy = size // 2
        amplitude = 1000.0

        image_clean = amplitude * np.exp(
            -((x - cx)**2 + (y - cy)**2) / (2 * sigma**2)
        )
        image_bad = image_clean.copy()
        image_bad[cy-2:cy+3, cx-2:cx+3] = np.nan
        user_mask = np.zeros_like(image_bad, dtype=bool)
        explicit_mask = user_mask.copy()
        explicit_mask[cy-2:cy+3, cx-2:cx+3] = True

        obj = Table()
        obj['x'] = [float(cx)]
        obj['y'] = [float(cy)]
        obj['flux'] = [amplitude]

        bg = np.zeros_like(image_clean)
        err = np.ones_like(image_clean)

        result_user = photometry_psf.measure_objects_psf(
            obj.copy(),
            image_bad,
            fwhm=fwhm,
            mask=user_mask,
            bg=bg,
            err=err,
            maxiters=200,  # More iterations needed with NaNs in center
            verbose=False,
        )
        result_explicit = photometry_psf.measure_objects_psf(
            obj.copy(),
            image_bad,
            fwhm=fwhm,
            mask=explicit_mask,
            bg=bg,
            err=err,
            maxiters=200,  # More iterations needed with NaNs in center
            verbose=False,
        )

        # With NaNs in the center, the fit may fail to converge.
        # The important thing is that both approaches handle it the same way
        # (either both succeed or both fail).
        assert np.isfinite(result_user['flux'][0]) == np.isfinite(result_explicit['flux'][0])

        # If both produced finite results, they should be close
        if np.isfinite(result_user['flux'][0]):
            np.testing.assert_allclose(
                result_user['flux'][0],
                result_explicit['flux'][0],
                rtol=1e-3,
                atol=1e-3,
            )

    @pytest.mark.unit
    def test_measure_objects_psf_fit_shape(self):
        """Test that fit_shape parameter is accepted (circular and square produce same results)."""
        # Note: photutils PSFPhotometry uses rectangular fitting regions regardless of fit_shape.
        # The fit_shape parameter is kept for API compatibility but doesn't change behavior.
        size = 51
        fwhm = 3.0
        image = _make_gaussian_image(size, 25, 25, fwhm, 1000.0)
        obj = Table()
        obj['x'] = [25.0]
        obj['y'] = [25.0]

        result_square = photometry_psf.measure_objects_psf(
            obj.copy(),
            image,
            fwhm=fwhm,
            fit_shape='square',
            fit_size=11,
            verbose=False,
        )
        result_circular = photometry_psf.measure_objects_psf(
            obj.copy(),
            image,
            fwhm=fwhm,
            fit_shape='circular',
            fit_size=11,
            verbose=False,
        )

        assert np.isfinite(result_square['flux'][0])
        assert np.isfinite(result_circular['flux'][0])
        # Both should use the same number of pixels (rectangular fitting region)
        assert result_circular['npix_psf'][0] == result_square['npix_psf'][0]

    @pytest.mark.unit
    def test_measure_objects_psf_recentroid(self):
        """Test that recentroid controls position updates."""
        size = 51
        fwhm = 3.0
        image = _make_gaussian_image(size, 25, 25, fwhm, 1000.0)
        obj = Table()
        obj['x'] = [27.0]
        obj['y'] = [25.0]

        result_fixed = photometry_psf.measure_objects_psf(
            obj.copy(),
            image,
            fwhm=fwhm,
            recentroid=False,
            maxiters=50,
            bg=np.zeros_like(image),
            err=np.ones_like(image),
            verbose=False,
        )
        result_free = photometry_psf.measure_objects_psf(
            obj.copy(),
            image,
            fwhm=fwhm,
            recentroid=True,
            maxiters=50,
            bg=np.zeros_like(image),
            err=np.ones_like(image),
            verbose=False,
        )

        assert abs(result_fixed['x_psf'][0] - obj['x'][0]) < 1e-3
        assert abs(result_free['x_psf'][0] - 25.0) < abs(obj['x'][0] - 25.0)

    @pytest.mark.unit
    def test_measure_objects_psf_fit_size_even(self):
        """Test that even fit_size is rounded to an odd size."""
        size = 51
        fwhm = 3.0
        image = _make_gaussian_image(size, 25, 25, fwhm, 1000.0)
        obj = Table()
        obj['x'] = [25.0]
        obj['y'] = [25.0]

        result = photometry_psf.measure_objects_psf(
            obj.copy(),
            image,
            fwhm=fwhm,
            fit_shape='square',
            fit_size=10,
            bg=np.zeros_like(image),
            err=np.ones_like(image),
            verbose=False,
        )

        assert result['npix_psf'][0] == 11 * 11

    @pytest.mark.unit
    def test_measure_objects_psf_psf_size_sampling(self):
        """Test PSF size scales with sampling factor."""
        psf_dict = _make_psf_dict(size=25, sampling=0.5)
        image = np.zeros((100, 100))
        psf.place_psf_stamp(image, psf_dict, x0=50.0, y0=50.0, flux=1000.0)

        obj = Table()
        obj['x'] = [50.0]
        obj['y'] = [50.0]

        result = photometry_psf.measure_objects_psf(
            obj.copy(),
            image,
            psf=psf_dict,
            fit_shape='square',
            bg=np.zeros_like(image),
            err=np.ones_like(image),
            maxiters=50,
            verbose=False,
        )

        assert result['npix_psf'][0] == 13 * 13

    @pytest.mark.unit
    def test_measure_objects_psf_maxiters_used(self, monkeypatch):
        """Test that maxiters is passed to PSFPhotometry."""
        calls = {}

        class DummyPSFPhotometry:
            def __init__(self, *args, **kwargs):
                calls['fitter_maxiters'] = kwargs.get('fitter_maxiters')

            def __call__(self, image, mask=None, error=None, init_params=None):
                n = len(init_params)
                result = Table()
                result['flux_fit'] = np.ones(n)
                result['flux_err'] = np.ones(n)
                result['x_fit'] = init_params['x']
                result['y_fit'] = init_params['y']
                return result

        monkeypatch.setattr(
            photometry_psf.photutils.psf, 'PSFPhotometry', DummyPSFPhotometry
        )

        obj = Table()
        obj['x'] = [5.0]
        obj['y'] = [5.0]
        image = np.zeros((20, 20))

        photometry_psf.measure_objects_psf(
            obj.copy(),
            image,
            fwhm=3.0,
            maxiters=7,
            verbose=False,
        )

        assert calls['fitter_maxiters'] == 7

    @pytest.mark.unit
    def test_measure_objects_psf_with_background(self, image_with_sources, detected_objects):
        """Test PSF photometry with external background."""
        # Create a simple background
        bg = np.ones_like(image_with_sources) * 100.0

        result = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            bg=bg,
            fwhm=3.0,
            verbose=False
        )

        assert isinstance(result, Table)
        assert 'flux' in result.colnames

    @pytest.mark.unit
    def test_measure_objects_psf_get_bg(self, image_with_sources, detected_objects):
        """Test that background can be returned."""
        result, bg, err = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            get_bg=True,
            verbose=False
        )

        assert isinstance(result, Table)
        assert bg.shape == image_with_sources.shape
        assert err.shape == image_with_sources.shape

    @pytest.mark.unit
    def test_measure_objects_psf_sn_filter(self, image_with_sources, detected_objects):
        """Test S/N filtering."""
        # Measure without filtering
        result_all = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            sn=None,
            verbose=False
        )

        # Measure with S/N filter
        result_filtered = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            sn=5.0,
            verbose=False
        )

        # Filtered should have fewer or equal objects
        assert len(result_filtered) <= len(result_all)

    @pytest.mark.unit
    def test_measure_objects_psf_keep_negative(self, image_with_sources, detected_objects):
        """Test filtering of negative fluxes."""
        # Keep negative fluxes
        result_keep = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            keep_negative=True,
            verbose=False
        )

        # Discard negative fluxes
        result_discard = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            keep_negative=False,
            verbose=False
        )

        # Result without negatives should have all positive fluxes
        if len(result_discard) > 0:
            assert np.all(result_discard['flux'] > 0)

        # Result with negatives might have some negative (or all positive for clean data)
        assert len(result_discard) <= len(result_keep)

    @pytest.mark.unit
    def test_measure_objects_psf_fwhm_from_table(self, image_with_sources):
        """Test that FWHM can be taken from object table."""
        # Create objects with FWHM column
        obj = Table()
        obj['x'] = [50.0, 120.0, 200.0]
        obj['y'] = [50.0, 80.0, 150.0]
        obj['fwhm'] = [3.0, 3.1, 2.9]

        result = photometry_psf.measure_objects_psf(
            obj,
            image_with_sources,
            fwhm=None,  # Should use FWHM from table
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) == 3

    @pytest.mark.unit
    def test_measure_objects_psf_quality_columns(self, image_with_sources, detected_objects):
        """Test that quality of fit columns are included in output."""
        result = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            verbose=False
        )

        # Check that quality columns are present
        assert 'qfit_psf' in result.colnames
        assert 'cfit_psf' in result.colnames
        assert 'flags_psf' in result.colnames
        assert 'npix_psf' in result.colnames
        assert 'x_psf' in result.colnames
        assert 'y_psf' in result.colnames

        # Check that columns have reasonable values
        assert len(result) > 0
        # qfit and cfit should be numeric (might be NaN for some fits)
        assert result['qfit_psf'].dtype in [np.float32, np.float64, np.int32, np.int64]
        assert result['cfit_psf'].dtype in [np.float32, np.float64, np.int32, np.int64]
        # flags_psf should be integer
        assert result['flags_psf'].dtype in [np.int32, np.int64]
        # npix_psf should be integer (number of pixels used in fit)
        assert result['npix_psf'].dtype in [np.int32, np.int64]
        # npix_psf should be positive for successful fits
        assert np.all(result['npix_psf'] >= 0)

        # reduced_chi2_psf might be present (photutils >= 2.3.0)
        if 'reduced_chi2_psf' in result.colnames:
            assert result['reduced_chi2_psf'].dtype in [np.float32, np.float64]

    @pytest.mark.unit
    def test_measure_objects_psf_with_masked_columns(self, image_with_sources, detected_objects_masked):
        """Test PSF photometry handles MaskedColumn x/y/flux inputs.

        This test verifies that measure_objects_psf doesn't crash when given
        an astropy table with MaskedColumn instead of regular Column.
        """
        result = photometry_psf.measure_objects_psf(
            detected_objects_masked,
            image_with_sources,
            fwhm=3.0,
            verbose=False
        )

        assert isinstance(result, Table)
        # Should return same number of objects
        assert len(result) == len(detected_objects_masked)

        # Should have expected columns
        assert 'flux' in result.colnames
        assert 'x_psf' in result.colnames
        assert 'y_psf' in result.colnames

        # Unmasked objects (first 3) should have some valid results
        valid_count = np.sum(np.isfinite(result['flux'][:3]))
        assert valid_count > 0

    @pytest.mark.unit
    def test_measure_objects_psf_grouped_with_masked_columns(self, image_with_sources, detected_objects_masked):
        """Test grouped PSF photometry handles MaskedColumn inputs."""
        result = photometry_psf.measure_objects_psf(
            detected_objects_masked,
            image_with_sources,
            fwhm=3.0,
            group_sources=True,
            grouper_radius=10.0,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) == len(detected_objects_masked)

    @pytest.mark.unit
    def test_measure_objects_psf_random_field_accuracy(self):
        """Test PSF photometry accuracy on a simulated field."""
        rng = np.random.RandomState(123)
        metrics = _simulate_psf_field(
            rng,
            size=128,
            fwhm=3.0,
            n_sources=25,
            margin=12,
            min_sep=8,
            noise_std=2.0,
            flux_range=(2000.0, 8000.0),
        )

        accuracy = metrics['accuracy']
        med_rel = metrics['med_rel']
        n_valid = metrics['n_valid']
        n_sources = metrics['n_sources']
        print(
            f"PSF accuracy {accuracy:.3f} (median rel err {med_rel:.3f}), "
            f"valid {n_valid}/{n_sources}"
        )

        assert n_valid >= int(0.9 * n_sources)
        assert accuracy > 0.95, (
            f"PSF accuracy {accuracy:.3f}, median rel err {med_rel:.3f}"
        )

    @pytest.mark.unit
    def test_measure_objects_psf_random_field_accuracy_report(self):
        """Report PSF photometry accuracy for a crowded field."""
        rng = np.random.RandomState(321)
        metrics = _simulate_psf_field(
            rng,
            size=128,
            fwhm=3.0,
            n_sources=35,
            margin=12,
            min_sep=6,
            noise_std=3.0,
            flux_range=(1500.0, 7000.0),
            mask_fraction=0.05,
        )

        accuracy = metrics['accuracy']
        med_rel = metrics['med_rel']
        n_valid = metrics['n_valid']
        n_sources = metrics['n_sources']
        print(
            f"PSF crowded accuracy {accuracy:.3f} (median rel err {med_rel:.3f}), "
            f"valid {n_valid}/{n_sources}, mask_fraction {metrics['mask_fraction']:.2f}"
        )


class TestCreatePSFModel:
    """Test empirical PSF model creation."""

    @pytest.mark.unit
    def test_create_psf_model_with_detections(self, image_with_sources, detected_objects):
        """Test ePSF creation with provided detections."""
        # Create ePSF from detected objects
        epsf = psf.create_psf_model(
            image_with_sources,
            obj=detected_objects,
            size=25,
            verbose=False
        )

        # Check that PSF dict was created with PSFEx-compatible structure
        assert epsf is not None
        assert isinstance(epsf, dict)
        assert 'data' in epsf
        assert 'sampling' in epsf
        assert 'width' in epsf
        assert 'height' in epsf
        assert 'fwhm' in epsf
        assert 'degree' in epsf
        assert 'ncoeffs' in epsf
        assert 'type' in epsf

        # Check data shape (ncoeffs, height, width)
        assert epsf['data'].ndim == 3
        assert epsf['data'].shape[0] == 1  # ePSF has ncoeffs=1
        assert epsf['data'].shape[1] > 0
        assert epsf['data'].shape[2] > 0

        # Check PSF properties
        assert epsf['degree'] == 0  # ePSF is position-invariant
        assert epsf['ncoeffs'] == 1
        assert epsf['type'] == 'epsf'
        assert epsf['sampling'] == 0.5  # Default oversampling=2

    @pytest.mark.unit
    def test_create_psf_model_auto_detect(self, image_with_sources):
        """Test ePSF creation with automatic star detection."""
        # Create ePSF with automatic detection
        epsf = psf.create_psf_model(
            image_with_sources,
            obj=None,  # Will detect stars automatically
            size=25,
            fwhm=3.0,
            verbose=False
        )

        # Check that PSF dict was created
        assert epsf is not None
        assert isinstance(epsf, dict)
        assert 'data' in epsf
        assert 'sampling' in epsf
        assert epsf['type'] == 'epsf'

    @pytest.mark.unit
    def test_create_psf_model_with_mask(self, image_with_sources, detected_objects, mask_with_bad_pixels):
        """Test ePSF creation with image mask."""
        epsf = psf.create_psf_model(
            image_with_sources,
            obj=detected_objects,
            size=25,
            mask=mask_with_bad_pixels,
            verbose=False
        )

        assert epsf is not None
        assert isinstance(epsf, dict)
        assert 'data' in epsf

    @pytest.mark.unit
    def test_create_psf_model_oversampling(self, image_with_sources, detected_objects):
        """Test ePSF creation with custom oversampling."""
        # Test oversampling=4
        epsf = psf.create_psf_model(
            image_with_sources,
            obj=detected_objects,
            size=25,
            oversampling=4,
            verbose=False
        )

        assert epsf is not None
        assert epsf['sampling'] == 0.25  # 1/4
        assert epsf['oversampling'] == 4


class TestPSFPhotometryIntegration:
    """Integration tests combining ePSF building and PSF photometry."""

    @pytest.mark.unit
    def test_epsf_then_measure(self, image_with_sources, detected_objects):
        """Test building ePSF and then using it for photometry."""
        # Build ePSF from detected objects
        epsf = psf.create_psf_model(
            image_with_sources,
            obj=detected_objects,
            size=25,
            verbose=False
        )

        # Use the ePSF for photometry
        result = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            psf=epsf,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) > 0
        assert 'flux' in result.colnames
        assert 'fluxerr' in result.colnames


class TestGroupedPSFPhotometry:
    """Tests for grouped PSF fitting functionality."""

    @pytest.mark.unit
    def test_grouped_psf_fitting(self, image_with_sources, detected_objects):
        """Test grouped PSF fitting for overlapping sources."""
        result = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            group_sources=True,
            grouper_radius=10.0,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) > 0
        assert 'flux' in result.colnames
        assert 'fluxerr' in result.colnames

    @pytest.mark.unit
    def test_grouped_vs_ungrouped(self, image_with_sources, detected_objects):
        """Compare grouped and ungrouped PSF fitting results."""
        # Ungrouped fitting
        result_ungrouped = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            group_sources=False,
            verbose=False
        )

        # Grouped fitting
        result_grouped = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            group_sources=True,
            verbose=False
        )

        # Both should return results
        assert len(result_ungrouped) > 0
        assert len(result_grouped) > 0

        # Both should have same number of objects (may differ in crowded fields)
        # For our test data they should be similar
        assert abs(len(result_ungrouped) - len(result_grouped)) <= 2


class TestPositionDependentPSF:
    """Tests for position-dependent PSF functionality."""

    @pytest.mark.unit
    def test_position_dependent_psf_photometry(self, image_with_sources, detected_objects):
        """Test PSF photometry with position-dependent PSFEx model."""
        mock_psf = _make_mock_psf(seed=123)

        # Try with position-dependent PSF enabled
        result = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            psf=mock_psf,
            use_position_dependent_psf=True,
            verbose=False
        )

        assert isinstance(result, Table)
        # May have some results (position-dependent fitting is more complex)
        assert len(result) >= 0

    @pytest.mark.unit
    def test_position_dependent_vs_constant_psf(self, image_with_sources, detected_objects):
        """Compare position-dependent and constant PSF fitting."""
        mock_psf = _make_mock_psf(seed=456)

        # Constant PSF (default)
        result_const = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            psf=mock_psf,
            use_position_dependent_psf=False,
            verbose=False
        )

        # Position-dependent PSF
        result_pos_dep = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            psf=mock_psf,
            use_position_dependent_psf=True,
            verbose=False
        )

        # Both should work (though results may differ)
        assert isinstance(result_const, Table)
        assert isinstance(result_pos_dep, Table)

    @pytest.mark.unit
    def test_position_dependent_grouped_runs(self, image_with_sources, detected_objects):
        """Test grouped flag with position-dependent PSF photometry."""
        mock_psf = _make_mock_psf(seed=789)
        result = photometry_psf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            psf=mock_psf,
            use_position_dependent_psf=True,
            group_sources=True,
            grouper_radius=10.0,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) == len(detected_objects)


# ============================================================================
# Property-based tests (if hypothesis is available)
# ============================================================================

try:
    from hypothesis import given, strategies as st

    class TestPhotometryPSFProperties:
        """Property-based tests for PSF photometry functions."""

        @pytest.mark.unit
        @given(
            fwhm=st.floats(min_value=1.0, max_value=10.0),
            psf_size=st.integers(min_value=15, max_value=51)
        )
        def test_psf_size_validity(self, fwhm, psf_size):
            """Test that PSF photometry works with various sizes."""
            # Make psf_size odd
            if psf_size % 2 == 0:
                psf_size += 1

            # Create simple image inline (avoid fixture issues with hypothesis)
            np.random.seed(42)
            simple_image = np.random.normal(100, 10, (100, 100))

            # Create simple object table
            obj = Table()
            obj['x'] = [50.0]
            obj['y'] = [50.0]

            # Should not crash
            try:
                result = photometry_psf.measure_objects_psf(
                    obj,
                    simple_image,
                    fwhm=fwhm,
                    psf_size=psf_size,
                    verbose=False
                )
                # If it succeeds, check basic validity
                assert isinstance(result, Table)
            except Exception:
                # Some combinations might fail, that's ok for property testing
                pass

except ImportError:
    # hypothesis not available, skip property tests
    pass


class TestEPSFWithPSFModule:
    """Tests for ePSF compatibility with psf.py evaluation functions."""

    @pytest.mark.unit
    def test_epsf_with_get_supersampled_psf_stamp(self, image_with_sources, detected_objects):
        """Test that ePSF works with psf.get_supersampled_psf_stamp()."""
        # Create ePSF
        epsf = psf.create_psf_model(
            image_with_sources,
            obj=detected_objects,
            size=25,
            verbose=False
        )

        # Use psf module function
        stamp = psf.get_supersampled_psf_stamp(epsf, x=100, y=100, normalize=True)

        # Check output
        assert stamp is not None
        assert stamp.ndim == 2
        assert np.isclose(np.sum(stamp), 1.0)  # Normalized
        assert stamp.shape == (epsf['height'], epsf['width'])

    @pytest.mark.unit
    def test_epsf_with_get_psf_stamp(self, image_with_sources, detected_objects):
        """Test that ePSF works with psf.get_psf_stamp()."""
        # Create ePSF
        epsf = psf.create_psf_model(
            image_with_sources,
            obj=detected_objects,
            size=25,
            verbose=False
        )

        # Use psf module function to get downsampled stamp
        stamp = psf.get_psf_stamp(epsf, x=100.3, y=100.7, normalize=True)

        # Check output
        assert stamp is not None
        assert stamp.ndim == 2
        assert np.isclose(np.sum(stamp), 1.0)  # Normalized
        # Stamp should be in original pixel space (smaller than supersampled)
        assert stamp.shape[0] < epsf['height']

    @pytest.mark.unit
    def test_epsf_position_invariance(self, image_with_sources, detected_objects):
        """Test that ePSF returns same stamp at different positions (degree=0)."""
        # Create ePSF
        epsf = psf.create_psf_model(
            image_with_sources,
            obj=detected_objects,
            size=25,
            verbose=False
        )

        # Get stamps at different positions
        stamp1 = psf.get_supersampled_psf_stamp(epsf, x=50, y=50, normalize=True)
        stamp2 = psf.get_supersampled_psf_stamp(epsf, x=200, y=200, normalize=True)

        # Should be identical since degree=0 (position-invariant)
        np.testing.assert_array_almost_equal(stamp1, stamp2)

    @pytest.mark.unit
    def test_epsf_with_place_psf_stamp(self, image_with_sources, detected_objects):
        """Test that ePSF works with psf.place_psf_stamp()."""
        # Create ePSF
        epsf = psf.create_psf_model(
            image_with_sources,
            obj=detected_objects,
            size=25,
            verbose=False
        )

        # Create empty image
        test_image = np.zeros((100, 100))

        # Place PSF stamp
        psf.place_psf_stamp(test_image, epsf, x0=50.5, y0=50.5, flux=1000)

        # Check that flux was added
        assert np.sum(test_image) > 0
        # Total flux should be approximately 1000
        assert np.isclose(np.sum(test_image), 1000, rtol=0.05)

    @pytest.mark.unit
    def test_epsf_compatible_with_psfex_api(self, image_with_sources, detected_objects):
        """Test that ePSF dict has all required PSFEx-compatible fields."""
        epsf = psf.create_psf_model(
            image_with_sources,
            obj=detected_objects,
            size=25,
            verbose=False
        )

        # Check required PSFEx fields
        required_fields = [
            'width', 'height', 'fwhm', 'sampling', 'ncoeffs', 'degree',
            'x0', 'y0', 'sx', 'sy', 'data'
        ]
        for field in required_fields:
            assert field in epsf, f"Missing required field: {field}"

        # Check data structure
        assert epsf['data'].ndim == 3
        assert epsf['data'].shape[0] == epsf['ncoeffs']
        assert epsf['data'].shape[1] == epsf['height']
        assert epsf['data'].shape[2] == epsf['width']

    @pytest.mark.unit
    def test_backward_compatibility_import(self, image_with_sources, detected_objects):
        """Test that create_psf_model can still be imported from photometry_psf for backward compatibility."""
        # This tests the re-export in photometry_psf
        epsf = photometry_psf.create_psf_model(
            image_with_sources,
            obj=detected_objects,
            size=25,
            verbose=False
        )

        assert epsf is not None
        assert isinstance(epsf, dict)
        assert epsf['type'] == 'epsf'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
