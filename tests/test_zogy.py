"""
Unit tests for ZOGY image subtraction (stdpipe.subtraction.run_zogy).
"""

import pytest
import numpy as np

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
