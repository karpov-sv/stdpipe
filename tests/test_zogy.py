"""
Unit tests for ZOGY image subtraction (stdpipe.subtraction.run_zogy).
"""

import pytest
import numpy as np
from astropy.table import Table

from stdpipe import psf, simulation, subtraction


def _image_pair(shape=(96, 96), seed=1):
    rng = np.random.default_rng(seed)
    model = simulation.create_psf_model(fwhm=3.0, psf_type='gaussian')

    image = rng.normal(100, 3, shape)
    template = rng.normal(100, 3, shape)
    for x, y, flux in [(30, 30, 5000), (60, 40, 3000), (45, 70, 8000)]:
        psf.place_psf_stamp(image, model, x, y, flux=flux)
        psf.place_psf_stamp(template, model, x, y, flux=flux)

    return image, template, model


class TestZogyScale:
    """Test the handling of the initial template flux scale."""

    @pytest.mark.unit
    @pytest.mark.parametrize('scale', [np.nan, np.inf, 0.0, -1.0])
    def test_unusable_initial_scale(self, scale):
        """least_squares used to fail on NaN or infinite initial scale."""
        image, template, model = _image_pair()
        err = np.full(image.shape, 3.0)

        diff = subtraction.run_zogy(
            image, template, err=err, template_err=err,
            image_psf=model, template_psf=model, scale=scale,
            fit_scale=True, fit_shift=True, verbose=False,
        )

        assert np.all(np.isfinite(diff))

    @pytest.mark.unit
    def test_initial_scale_ignores_bad_magnitude_errors(self):
        """Zero or NaN magnitude errors used to make the estimated scale NaN."""
        image, template, model = _image_pair()
        err = np.full(image.shape, 3.0)

        rng = np.random.default_rng(2)
        n = 10
        x = rng.uniform(20, 76, n)
        y = rng.uniform(20, 76, n)
        mag = rng.uniform(14, 18, n)
        magerr = rng.uniform(0.01, 0.05, n)
        image_obj = Table({'x': x, 'y': y, 'flags': np.zeros(n, int), 'mag': mag, 'magerr': magerr})
        # Template twice brighter, with some unusable errors
        template_obj = image_obj.copy()
        template_obj['mag'] = mag - 2.5 * np.log10(2.0)
        template_obj['magerr'][[0, 3]] = 0.0
        template_obj['magerr'][5] = np.nan

        messages = []
        subtraction.run_zogy(
            image, template, err=err, template_err=err,
            image_psf=model, template_psf=model,
            image_obj=image_obj, template_obj=template_obj,
            fit_scale=False, verbose=lambda *args: messages.append(' '.join(map(str, args))),
        )

        estimated = [_ for _ in messages if 'Estimated initial template flux scale' in _]
        assert len(estimated) == 1
        assert float(estimated[0].split('=')[-1]) == pytest.approx(2.0, rel=0.01)
