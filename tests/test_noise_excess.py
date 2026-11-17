"""Tests for photometry_background.estimate_noise_excess()."""

import numpy as np
import pytest
from scipy import ndimage
from astropy.table import Table

from stdpipe.photometry_background import estimate_noise_excess, _noise_excess_kernel
from stdpipe import photometry_measure

pytestmark = pytest.mark.unit


@pytest.fixture(scope='module')
def white():
    return np.random.default_rng(0).normal(0, 1, (512, 512))


@pytest.fixture(scope='module')
def shifted(white):
    """Bilinear half-pixel shift: per-pixel variance 1/4, sums over large
    apertures keep the full variance, so the excess tends to 2."""
    img = ndimage.shift(white, (0.5, 0.5), order=1)
    return img, np.std(img[2:-2, 2:-2])


def _gaussian_psf_dict(sigma, size=31):
    half = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    g = np.exp(-((xx - half) ** 2 + (yy - half) ** 2) / (2 * sigma**2))
    g /= g.sum()
    return {
        'data': g[None, :, :], 'width': size, 'height': size,
        'sampling': 1.0, 'degree': 0, 'ncoeffs': 1,
        'x0': 0.0, 'y0': 0.0, 'sx': 1.0, 'sy': 1.0, 'type': 'epsf',
    }


def test_white_noise(white):
    excess = estimate_noise_excess(white, 1.0, bg=0, aper=[1, 3, 8])
    assert excess.shape == (3,)
    assert np.allclose(excess, 1.0, atol=0.05)


def test_correlated_noise_grows_with_aperture(shifted):
    img, rms = shifted
    excess, stats = estimate_noise_excess(img, rms, bg=0, aper=[0.5, 2, 8], get_stats=True)
    # A sub-pixel aperture covers one pixel - its noise is the per-pixel one
    assert excess[0] == pytest.approx(1.0, abs=0.05)
    assert np.all(np.diff(excess) > 0)
    # Only ~1000 disjoint r=8 apertures fit into the image
    assert stats[2]['excess_err'] > 0.03
    assert excess[-1] == pytest.approx(2.0, abs=3 * stats[2]['excess_err'])


def test_optimal_weights(white, shifted):
    img, rms = shifted
    sigma = 1.5
    psf = _gaussian_psf_dict(sigma)
    # Gaussian weights from fwhm and from the equivalent PSF model agree
    e_fwhm = estimate_noise_excess(img, rms, bg=0, aper=2, fwhm=2.355 * sigma, optimal=True)
    e_psf = estimate_noise_excess(img, rms, bg=0, aper=2, fwhm=2.355 * sigma, optimal=True, psf=psf)
    assert e_psf == pytest.approx(e_fwhm, rel=0.02)
    assert 1.5 < e_fwhm < 2.0
    assert estimate_noise_excess(white, 1.0, bg=0, aper=2, fwhm=3.0, optimal=True) == pytest.approx(
        1.0, abs=0.06
    )
    with pytest.raises(ValueError):
        estimate_noise_excess(img, rms, bg=0, optimal=True)


def test_aper_in_fwhm_units(shifted):
    img, rms = shifted
    e1 = estimate_noise_excess(img, rms, bg=0, aper=1.0, fwhm=4.0)
    e2 = estimate_noise_excess(img, rms, bg=0, aper=4.0)
    assert e1 == e2


def test_spatially_varying_err(white):
    """Each sum is normalized by its own predicted noise."""
    err = np.ones_like(white)
    err[:, 256:] = 5.0
    excess = estimate_noise_excess(white * err, err, bg=0, aper=3)
    assert excess == pytest.approx(1.0, abs=0.05)


def test_mask_excludes_sources_and_auto_background(white):
    n = white.shape[0]
    rng = np.random.default_rng(1)
    image = white + 100.0
    yy, xx = np.mgrid[:n, :n]
    mask = np.zeros(image.shape, bool)
    for x0, y0 in rng.uniform(0, n, (150, 2)):
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        image += 300.0 * np.exp(-r2 / 2 / 1.5**2)
        mask |= r2 < 8**2

    excess, stats = estimate_noise_excess(image, mask=mask, aper=3, get_stats=True)
    assert excess == pytest.approx(1.0, abs=0.05)
    assert stats['nsamples'] == 5000
    assert abs(stats['median']) < 0.1
    assert 0 < stats['excess_err'] < 0.05

    # Unmasked sources inflate the scatter
    assert estimate_noise_excess(image, aper=3) > excess


def test_non_finite_pixels_rejected(white):
    image = white.copy()
    image[100:200, 100:200] = np.nan
    err = np.ones_like(white)
    err[300:400, 300:400] = np.nan
    # Stabilize the sampled MAD after excluding both non-finite regions.
    excess = estimate_noise_excess(image, err, bg=0, aper=3, nsamples=10000)
    assert np.isfinite(excess)
    assert excess == pytest.approx(1.0, abs=0.05)


def test_too_few_apertures(white):
    mask = np.ones(white.shape, bool)
    excess, stats = estimate_noise_excess(white, 1.0, bg=0, mask=mask, aper=3, get_stats=True)
    assert np.isnan(excess)
    assert stats['nsamples'] == 0


@pytest.mark.parametrize('optimal', [False, True])
def test_corrected_photometry_errors(optimal):
    """Errors of faint stars on a resampled image match their actual scatter
    once the background noise map is scaled by the excess."""
    rng = np.random.default_rng(3)
    n, sigma, flux = 512, 1.5, 300.0
    noise = ndimage.shift(rng.normal(0, 10, (n, n)), (0.4, 0.3), order=3)
    noise = ndimage.gaussian_filter(noise, 0.7)
    rms = np.std(noise)

    xs, ys = np.meshgrid(np.arange(30, n - 30, 25) + 0.2, np.arange(30, n - 30, 25) + 0.3)
    xs, ys = xs.ravel(), ys.ravel()
    image = noise + 100.0
    mask = np.zeros(image.shape, bool)
    yy, xx = np.mgrid[-10:11, -10:11]
    for x0, y0 in zip(xs, ys):
        ix, iy = int(x0), int(y0)
        image[iy - 10 : iy + 11, ix - 10 : ix + 11] += (
            flux / (2 * np.pi * sigma**2)
            * np.exp(-((xx + ix - x0) ** 2 + (yy + iy - y0) ** 2) / 2 / sigma**2)
        )
        mask[iy - 7 : iy + 8, ix - 7 : ix + 8] = True

    kw = dict(aper=1.0, fwhm=2.355 * sigma, optimal=optimal, bg=100.0)
    excess = estimate_noise_excess(image, rms, mask=mask, **kw)
    assert excess > 2

    obj = Table({'x': xs, 'y': ys})
    for bg_err, ok in ((rms, False), (excess * rms, True)):
        res = photometry_measure.measure_objects(
            obj, image, bg_err=bg_err, sn=0, group_sources=False, **kw
        )
        pull = np.std((res['flux'] - flux) / res['fluxerr'])
        assert (abs(pull - 1) < 0.08) == ok, pull


def test_measured_matches_kernel_and_sees_annulus(white):
    """Measured with the photometry itself, the excess agrees with the kernel one for
    plain apertures, and accounts for the noise of the local background estimate if the
    photometry does not."""
    image = 10 * white
    kw = dict(aper=4.0, bg=0.0, sn=0, verbose=False)

    def excess(drop_bg_term=False, **extra):
        def measure(o):
            res = photometry_measure.measure_objects(o, image, bg_err=10.0, **kw, **extra)
            if drop_bg_term:
                # Photometry errors without the noise of the local background
                res['fluxerr'] = np.sqrt(res['fluxerr']**2 - (res['npix_aper'] * res['bg_local_err'])**2)
            return res
        return estimate_noise_excess(image, aper=4.0, measure=measure)

    plain = excess()
    assert plain == pytest.approx(estimate_noise_excess(image, 10.0, bg=0, aper=4.0), abs=0.04)

    # measure_objects accounts for the noise of the local background itself
    for est in ['mean', 'mode']:
        assert excess(bkgann=(12, 20), bg_estimator=est) == pytest.approx(plain, abs=0.05)

    # ..and if it did not, the excess would: area ratio for the mean, much more for the mode
    mean = excess(True, bkgann=(12, 20), bg_estimator='mean')
    mode = excess(True, bkgann=(12, 20), bg_estimator='mode')
    assert plain < mean < mode
    assert mean / plain == pytest.approx(np.sqrt(1 + 16 / (400 - 144)), abs=0.04)
    assert mode / plain > 1.04


def test_measured_positions(white):
    mask = np.zeros(white.shape, bool)
    mask[:, :256] = True
    seen = []

    def measure(obj):
        seen.append(obj)
        return Table({'flux': np.zeros(len(obj)) + white[0, 0], 'fluxerr': np.ones(len(obj))})

    estimate_noise_excess(white, mask=mask, aper=3.0, spacing=20, measure=measure, nsamples=100)
    obj = seen[0]
    assert len(obj) == 100
    # Apertures do not touch the mask or the edges
    assert np.all(obj['x'] - 3 > 256 - 1) and np.all(obj['x'] + 3 < 511)
    # Positions are farther apart than the spacing minus the jitter
    from scipy.spatial import cKDTree
    d = cKDTree(np.column_stack([obj['x'], obj['y']])).query(np.column_stack([obj['x'], obj['y']]), k=2)[0][:, 1]
    assert d.min() > 19

    with pytest.raises(ValueError):
        estimate_noise_excess(white, aper=[1, 2], measure=measure)


def test_measured_failures(white):
    # Too few positions, and too few properly measured ones
    assert np.isnan(estimate_noise_excess(white, mask=np.ones(white.shape, bool), aper=3,
                                          measure=lambda o: None))
    nan = lambda o: Table({'flux': np.full(len(o), np.nan), 'fluxerr': np.ones(len(o))})
    excess, stats = estimate_noise_excess(white, aper=3, measure=nan, get_stats=True)
    assert np.isnan(excess) and stats['nsamples'] == 0


def test_corrected_photometry_errors_with_annulus():
    """With mode local background, the excess measured by the photometry itself
    makes the errors of faint stars on a resampled image match their scatter."""
    rng = np.random.default_rng(4)
    n, sigma, flux = 768, 1.5, 300.0
    noise = ndimage.gaussian_filter(ndimage.shift(rng.normal(0, 10, (n, n)), (0.4, 0.3), order=3), 0.7)
    rms = np.std(noise)
    xs, ys = np.meshgrid(np.arange(40, n - 40, 30) + 0.2, np.arange(40, n - 40, 30) + 0.3)
    xs, ys = xs.ravel(), ys.ravel()
    image = noise + 100.0
    mask = np.zeros(image.shape, bool)
    yy, xx = np.mgrid[-10:11, -10:11]
    for x0, y0 in zip(xs, ys):
        ix, iy = int(x0), int(y0)
        image[iy - 10 : iy + 11, ix - 10 : ix + 11] += (
            flux / (2 * np.pi * sigma**2)
            * np.exp(-((xx + ix - x0) ** 2 + (yy + iy - y0) ** 2) / 2 / sigma**2)
        )
        mask[iy - 7 : iy + 8, ix - 7 : ix + 8] = True

    kw = dict(aper=1.0, fwhm=2.355 * sigma, bg=100.0, bkgann=(2.5, 4), bg_estimator='mode',
              mask=mask, sn=0, verbose=False)
    excess = estimate_noise_excess(
        image, mask=mask, aper=1.0, fwhm=2.355 * sigma,
        measure=lambda o: photometry_measure.measure_objects(o, image, bg_err=rms, **kw),
    )
    res = photometry_measure.measure_objects(Table({'x': xs, 'y': ys}), image,
                                             bg_err=excess * rms, **kw)
    pull = np.std((res['flux'] - flux) / res['fluxerr'])
    assert pull == pytest.approx(1.0, abs=0.08)
