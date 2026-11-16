"""
Tests for HOTPANTS wrapper and its post-run stamp check.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from stdpipe import subtraction
from stdpipe.subtraction import (
    _read_hotpants_stamps,
    _read_hotpants_regions,
    _measure_flux_ratios,
    _check_hotpants_stamps,
)


def _render(x, y, flux, fwhm, shape):
    """Image of Gaussian stars."""
    sigma = fwhm / 2.355
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    image = np.zeros(shape)
    for xi, yi, fi in zip(x, y, flux):
        sl = (
            slice(max(0, int(yi) - 30), int(yi) + 31),
            slice(max(0, int(xi) - 30), int(xi) + 31),
        )
        r2 = (xx[sl] - xi) ** 2 + (yy[sl] - yi) ** 2
        image[sl] += fi / (2 * np.pi * sigma**2) * np.exp(-r2 / 2 / sigma**2)
    return image


def _make_field(saturation=None, seed=1, shape=(400, 400), scale=0.1):
    """Science image with FWHM=5 and template with FWHM=2 scaled by `scale`, with a
    very bright star in the center that may be clipped at `saturation` in the template."""
    rng = np.random.default_rng(seed)
    n = 60
    x = np.r_[rng.uniform(30, shape[1] - 30, n), 200.3]
    y = np.r_[rng.uniform(30, shape[0] - 30, n), 200.7]
    flux = np.r_[10 ** rng.uniform(3.3, 4.7, n), 3e6]

    image = _render(x, y, flux, 5.0, shape) + rng.normal(0, 10, shape)
    template = _render(x, y, flux, 2.0, shape) * scale
    if saturation is not None:
        template = np.minimum(template, saturation)
    template += rng.normal(0, 0.3, shape)

    return image, template, Table({'x': x, 'y': y, 'flux': flux})


def _write_stamps(filename, xs, ys, size=47):
    with open(filename, 'w') as f:
        f.write('#   X    Y\n')
        for x, y in zip(xs, ys):
            f.write('image;box(%4d,%4d,%d,%d) # color=green\n' % (x + 1, y + 1, size, size))


@pytest.mark.unit
class TestStampCheckHelpers:
    def test_read_stamps(self, tmp_path):
        fname = str(tmp_path / 'stamps.reg')
        _write_stamps(fname, [10, 200], [20, 300])
        xs, ys, hws = _read_hotpants_stamps(fname)
        assert np.allclose(xs, [10, 200])
        assert np.allclose(ys, [20, 300])
        assert np.allclose(hws, 23)

    def test_read_stamps_missing(self, tmp_path):
        xs, ys, hws = _read_hotpants_stamps(str(tmp_path / 'nonexistent.reg'))
        assert len(xs) == 0

    def test_read_regions(self):
        header = fits.Header()
        header['NREGION'] = 2
        header['REGION00'] = '[1:200,1:400]'
        header['KSUM00'] = '10.5'
        header['REGION01'] = '[201:400,1:400]'
        header['KSUM01'] = 11.0
        regions = _read_hotpants_regions(header)
        assert regions == [(0, 199, 0, 399, 10.5), (200, 399, 0, 399, 11.0)]

    def test_flux_ratios(self):
        image, template, obj = _make_field()
        good = np.ones(image.shape, dtype=bool)
        ratio, ratio_err, sn1, sn2, _ = _measure_flux_ratios(
            image, template, good, obj['x'][-1:], obj['y'][-1:], 15
        )
        assert ratio[0] == pytest.approx(10, rel=0.01)
        assert sn1[0] > 100 and sn2[0] > 100

    def test_flux_ratios_masked_aperture(self):
        image, template, obj = _make_field()
        good = np.ones(image.shape, dtype=bool)
        good[180:220, 180:220] = False
        ratio, _, _, _, _ = _measure_flux_ratios(
            image, template, good, obj['x'][-1:], obj['y'][-1:], 15
        )
        assert np.isnan(ratio[0])

    def _check(self, tmp_path, image, template, ksum, stamps):
        fname = str(tmp_path / 'stamps.reg')
        _write_stamps(fname, stamps['x'], stamps['y'])
        header = fits.Header()
        header['NREGION'] = 1
        header['REGION00'] = '[1:%d,1:%d]' % (image.shape[1], image.shape[0])
        header['KSUM00'] = ksum
        good = np.ones(image.shape, dtype=bool)
        return _check_hotpants_stamps(
            image, template, good, fname, header, obj=stamps, aper=15, log=lambda *a: None
        )

    def test_check_consistent(self, tmp_path):
        image, template, obj = _make_field()
        assert self._check(tmp_path, image, template, 10.0, obj) == []

    def test_check_finds_culprit(self, tmp_path):
        image, template, obj = _make_field(saturation=2000)
        positions = self._check(tmp_path, image, template, 30.0, obj)
        assert len(positions) == 1
        px, py = positions[0]
        assert abs(px - 200) <= 1 and abs(py - 201) <= 1

    def test_check_ignores_opposite_outliers(self, tmp_path):
        # The kernel sum is low, while the clipped star has high ratio - it cannot be the cause
        image, template, obj = _make_field(saturation=2000)
        positions = self._check(tmp_path, image, template, 5.0, obj)
        assert all(abs(px - 200) > 5 or abs(py - 201) > 5 for px, py in positions)


@pytest.mark.integration
@pytest.mark.requires_hotpants
class TestHotpantsStampCheck:
    def _run(self, saturation, check):
        image, template, obj = _make_field(saturation=saturation)
        diff, header = subtraction.run_hotpants(
            image,
            template,
            image_fwhm=5,
            template_fwhm=2,
            image_gain=1000,
            template_gain=1e6,
            err=True,
            obj=obj,
            check_stamps=check,
            get_header=True,
        )
        return diff, header

    def test_clean_field_untouched(self):
        diff, header = self._run(None, True)
        assert header['NSTMASK'] == 0
        assert float(header['KSUM00']) == pytest.approx(10, rel=0.02)

    def test_saturated_template_star(self):
        _, header = self._run(2000, False)
        # Without the check, the clipped star drives the kernel normalization
        assert float(header['KSUM00']) > 15
        assert 'NSTMASK' not in header

        diff, header = self._run(2000, True)
        assert header['NSTMASK'] >= 1
        assert float(header['KSUM00']) == pytest.approx(10, rel=0.02)
        # The culprit is masked in the difference image
        assert diff[201, 200] == 1e-30
