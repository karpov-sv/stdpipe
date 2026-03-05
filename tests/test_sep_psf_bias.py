"""
Bias characterization tests for SEP-based PSF fitting photometry.

Tests photometric bias of measure_objects_sep(psf=...) under various conditions:
- Analytic PSF models (Gaussian, Moffat) at different FWHM and S/N
- PSF model mismatch (Gaussian model on Moffat stars)
- Sub-pixel position accuracy
- Grouped vs ungrouped fitting in crowded fields
- Empirical PSFEx models
- Comparison with photutils PSF photometry
"""

import pytest
import numpy as np
from scipy.special import erf
from astropy.table import Table

from stdpipe import simulation, photometry_measure
from stdpipe import psf as psf_module

# Check SEP PSF availability
_HAS_SEP_PSF = False
try:
    import sep

    if hasattr(sep, 'PSF') and hasattr(sep, 'psf_fit'):
        _HAS_SEP_PSF = True
except ImportError:
    pass

_skip_no_sep_psf = pytest.mark.skipif(
    not _HAS_SEP_PSF,
    reason="Requires SEP 1.4+ with psf_fit() and PSF class",
)


def _create_pixel_integrated_gaussian(size, x0, y0, flux, sigma):
    """Create pixel-integrated Gaussian image with given total flux."""
    x_edges = np.arange(size + 1) - 0.5
    y_edges = np.arange(size + 1) - 0.5
    sqrt2_sigma = np.sqrt(2) * sigma
    cdf_x = 0.5 * (1 + erf((x_edges - x0) / sqrt2_sigma))
    cdf_y = 0.5 * (1 + erf((y_edges - y0) / sqrt2_sigma))
    image = flux * np.outer(np.diff(cdf_y), np.diff(cdf_x))
    return image


def _place_stars(image, xs, ys, fluxes, psf_model):
    """Place stars at given positions using PSF model."""
    for x, y, flux in zip(xs, ys, fluxes):
        psf_module.place_psf_stamp(image, psf_model, x, y, flux=flux)


def _make_star_field(n_stars, img_size, edge, fwhm, fluxes, psf_type='gaussian',
                     beta=2.5, background=100.0, noise_std=1.0, seed=42):
    """Create a star field image with known positions and fluxes.

    Returns (image, obj_table) where obj_table has x, y, flux_input columns.
    """
    rng = np.random.RandomState(seed)
    xs = rng.uniform(edge, img_size - edge, n_stars)
    ys = rng.uniform(edge, img_size - edge, n_stars)

    image = np.full((img_size, img_size), background, dtype=np.float64)
    image += rng.normal(0, noise_std, image.shape)

    # Large PSF model for placing stars (no truncation)
    psf_model = simulation.create_psf_model(
        fwhm, psf_type=psf_type, beta=beta, size=101, oversampling=2,
    )
    _place_stars(image, xs, ys, fluxes, psf_model)

    obj = Table({
        'x': xs,
        'y': ys,
        'flux_input': np.asarray(fluxes, dtype=float),
    })
    return image, obj


def _measure_bias(image, obj, psf_arg, fwhm=4.0, gain=1e6, **kwargs):
    """Run SEP PSF photometry and return bias statistics.

    Returns dict with keys: med_bias, mean_bias, std_bias, n_good.
    Bias is (measured/true - 1) * 100 in percent.
    """
    result = photometry_measure.measure_objects_sep(
        obj.copy(), image, aper=1.5, fwhm=fwhm,
        psf=psf_arg, gain=gain, sn=0, verbose=False,
        **kwargs,
    )
    good = result[result['flags'] == 0]
    if len(good) == 0:
        return {'med_bias': np.nan, 'mean_bias': np.nan,
                'std_bias': np.nan, 'n_good': 0, 'result': result}
    bias = (good['flux'] / good['flux_input'] - 1) * 100
    return {
        'med_bias': np.median(bias),
        'mean_bias': np.mean(bias),
        'std_bias': np.std(bias),
        'n_good': len(good),
        'result': result,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analytic PSF models — no external dependencies
# ═══════════════════════════════════════════════════════════════════════════

@_skip_no_sep_psf
class TestSEPPSFBiasAnalytic:
    """Bias tests using analytic (Gaussian/Moffat) PSF models."""

    @pytest.mark.slow
    @pytest.mark.parametrize("fwhm", [1.5, 2, 3, 4, 6, 8])
    def test_flux_bias_vs_fwhm_gaussian(self, fwhm):
        """Gaussian stars + Gaussian PSF model: bias < 5% for all FWHM."""
        n_stars = 50
        img_size = max(256, int(fwhm * 30))
        edge = max(30, int(fwhm * 8))
        fluxes = np.full(n_stars, 10000.0)

        image, obj = _make_star_field(
            n_stars, img_size, edge, fwhm, fluxes,
            psf_type='gaussian', noise_std=1.0,
        )
        psf_model = simulation.create_psf_model(
            fwhm, psf_type='gaussian', size=max(33, int(fwhm * 8) + 1),
            oversampling=2,
        )
        stats = _measure_bias(image, obj, psf_model, fwhm=fwhm)

        assert stats['n_good'] > n_stars * 0.8, (
            f"Too few good sources: {stats['n_good']}/{n_stars}"
        )
        assert abs(stats['med_bias']) < 5.0, (
            f"FWHM={fwhm}: median bias {stats['med_bias']:.2f}% exceeds 5%"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("flux", [300, 1000, 3000, 10000, 100000])
    def test_flux_bias_vs_sn(self, flux):
        """Gaussian stars at various S/N: bias < 5% for S/N > ~10."""
        fwhm = 4.0
        n_stars = 50
        noise_std = 10.0  # higher noise for meaningful S/N variation

        image, obj = _make_star_field(
            n_stars, 256, 40, fwhm,
            np.full(n_stars, float(flux)),
            psf_type='gaussian', noise_std=noise_std,
        )
        psf_model = simulation.create_psf_model(
            fwhm, psf_type='gaussian', size=33, oversampling=2,
        )
        stats = _measure_bias(image, obj, psf_model, fwhm=fwhm)

        # Rough S/N estimate: flux / (noise_std * sqrt(pi * (2*fwhm)^2))
        approx_sn = flux / (noise_std * np.sqrt(np.pi * (2 * fwhm) ** 2))
        if approx_sn > 10 and stats['n_good'] > 10:
            assert abs(stats['med_bias']) < 5.0, (
                f"flux={flux}, ~S/N={approx_sn:.0f}: "
                f"median bias {stats['med_bias']:.2f}% exceeds 5%"
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("beta", [2.0, 2.5, 3.0, 4.5])
    def test_moffat_gaussian_mismatch(self, beta):
        """Moffat stars + Gaussian PSF: known mismatch, bias < 40%."""
        fwhm = 4.0
        n_stars = 50
        fluxes = np.full(n_stars, 10000.0)

        image, obj = _make_star_field(
            n_stars, 256, 40, fwhm, fluxes,
            psf_type='moffat', beta=beta, noise_std=1.0,
        )
        # Fit with Gaussian model (deliberate mismatch)
        psf_model = simulation.create_psf_model(
            fwhm, psf_type='gaussian', size=33, oversampling=2,
        )
        stats = _measure_bias(image, obj, psf_model, fwhm=fwhm)

        assert stats['n_good'] > n_stars * 0.8
        # Known mismatch — Gaussian underestimates Moffat wings
        assert abs(stats['med_bias']) < 40.0, (
            f"beta={beta}: median bias {stats['med_bias']:.2f}% exceeds 40%"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("beta", [2.0, 2.5, 3.0, 4.5])
    def test_moffat_matching_model(self, beta):
        """Moffat stars + matching Moffat PSF model: bias < 5%."""
        fwhm = 4.0
        n_stars = 50
        fluxes = np.full(n_stars, 10000.0)

        image, obj = _make_star_field(
            n_stars, 256, 40, fwhm, fluxes,
            psf_type='moffat', beta=beta, noise_std=1.0,
        )
        psf_model = simulation.create_psf_model(
            fwhm, psf_type='moffat', beta=beta, size=51, oversampling=2,
        )
        stats = _measure_bias(image, obj, psf_model, fwhm=fwhm)

        assert stats['n_good'] > n_stars * 0.8
        assert abs(stats['med_bias']) < 5.0, (
            f"beta={beta}: median bias {stats['med_bias']:.2f}% exceeds 5%"
        )

    @pytest.mark.slow
    def test_subpixel_position_accuracy(self):
        """Stars at sub-pixel offsets: position recovery < 0.1 pix."""
        fwhm = 4.0
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        size = 64
        flux = 50000.0
        offsets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        for dx in offsets:
            for dy in [0.0, 0.25, 0.5]:
                cx, cy = 32.0 + dx, 32.0 + dy
                image = _create_pixel_integrated_gaussian(size, cx, cy, flux, sigma)
                obj = Table({'x': [cx], 'y': [cy], 'flux_input': [flux]})

                psf = sep.PSF.from_gaussian(fwhm, size=25, oversampling=2)
                result = photometry_measure.measure_objects_sep(
                    obj.copy(), image, aper=1.5, fwhm=fwhm,
                    psf=psf, gain=1e6, sn=0, verbose=False,
                )
                good = result[result['flags'] == 0]
                assert len(good) == 1
                pos_err = np.sqrt(
                    (good['x_psf'][0] - cx) ** 2 + (good['y_psf'][0] - cy) ** 2
                )
                assert pos_err < 0.1, (
                    f"offset=({dx},{dy}): position error {pos_err:.4f} > 0.1 pix"
                )

    @pytest.mark.slow
    @pytest.mark.parametrize("separation", [2, 4, 6, 10, 20])
    def test_grouped_vs_ungrouped_crowded(self, separation):
        """Grouped fitting should be better or equal for close pairs."""
        fwhm = 4.0
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        size = 64
        flux = 10000.0

        cx1, cy = 32.0 - separation / 2.0, 32.0
        cx2 = 32.0 + separation / 2.0
        image = np.zeros((size, size), dtype=np.float64)
        image += _create_pixel_integrated_gaussian(size, cx1, cy, flux, sigma)
        image += _create_pixel_integrated_gaussian(size, cx2, cy, flux, sigma)

        obj = Table({
            'x': [cx1, cx2],
            'y': [cy, cy],
            'flux_input': [flux, flux],
        })
        psf = sep.PSF.from_gaussian(fwhm, size=25, oversampling=2)

        result_g = photometry_measure.measure_objects_sep(
            obj.copy(), image, aper=1.5, fwhm=fwhm,
            psf=psf, gain=1e6, sn=0, group_sources=True, verbose=False,
        )
        result_u = photometry_measure.measure_objects_sep(
            obj.copy(), image, aper=1.5, fwhm=fwhm,
            psf=psf, gain=1e6, sn=0, group_sources=False, verbose=False,
        )

        good_g = result_g[result_g['flags'] == 0]
        good_u = result_u[result_u['flags'] == 0]
        if len(good_g) == 2 and len(good_u) == 2:
            bias_g = np.mean(np.abs(good_g['flux'] / good_g['flux_input'] - 1))
            bias_u = np.mean(np.abs(good_u['flux'] / good_u['flux_input'] - 1))
            # Grouped should be at least as good
            assert bias_g <= bias_u * 1.1 + 0.01, (
                f"sep={separation}: grouped bias {bias_g:.3f} > ungrouped {bias_u:.3f}"
            )

    @pytest.mark.slow
    def test_statistical_bias_random_field(self):
        """Mean |bias| over 100 random stars should be < 3%."""
        fwhm = 4.0
        n_stars = 100
        rng = np.random.RandomState(12345)
        fluxes = rng.uniform(3000, 30000, n_stars)

        image, obj = _make_star_field(
            n_stars, 512, 40, fwhm, fluxes,
            psf_type='gaussian', noise_std=1.0, seed=12345,
        )
        psf_model = simulation.create_psf_model(
            fwhm, psf_type='gaussian', size=33, oversampling=2,
        )
        stats = _measure_bias(image, obj, psf_model, fwhm=fwhm)

        assert stats['n_good'] > 80
        assert abs(stats['mean_bias']) < 3.0, (
            f"Mean bias {stats['mean_bias']:.2f}% exceeds 3%"
        )
        assert abs(stats['med_bias']) < 3.0, (
            f"Median bias {stats['med_bias']:.2f}% exceeds 3%"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PSFEx empirical models — requires SExtractor + PSFEx
# ═══════════════════════════════════════════════════════════════════════════

@_skip_no_sep_psf
@pytest.mark.requires_psfex
@pytest.mark.requires_sextractor
class TestSEPPSFBiasPSFEx:
    """Bias tests using empirical PSFEx models."""

    @pytest.fixture
    def gaussian_field(self):
        """512x512 image with 80 Gaussian stars, FWHM=4."""
        fwhm = 4.0
        n_stars = 80
        fluxes = np.full(n_stars, 10000.0)
        image, obj = _make_star_field(
            n_stars, 512, 40, fwhm, fluxes,
            psf_type='gaussian', noise_std=1.0, seed=42,
        )
        psfex_dict = psf_module.run_psfex(image, thresh=3.0, order=0)
        return image, obj, psfex_dict, fwhm

    @pytest.fixture
    def moffat_field(self):
        """512x512 image with 80 Moffat (beta=2.5) stars, FWHM=4."""
        fwhm = 4.0
        n_stars = 80
        fluxes = np.full(n_stars, 10000.0)
        image, obj = _make_star_field(
            n_stars, 512, 40, fwhm, fluxes,
            psf_type='moffat', beta=2.5, noise_std=1.0, seed=42,
        )
        psfex_dict = psf_module.run_psfex(
            image, thresh=3.0, order=0, vignet_size=41,
        )
        return image, obj, psfex_dict, fwhm

    @pytest.mark.slow
    def test_psfex_gaussian_stars(self, gaussian_field):
        """PSFEx on Gaussian stars: bias < 5%."""
        image, obj, psfex_dict, fwhm = gaussian_field
        stats = _measure_bias(image, obj, psfex_dict, fwhm=fwhm)

        assert stats['n_good'] > 60
        assert abs(stats['med_bias']) < 5.0, (
            f"PSFEx on Gaussian: median bias {stats['med_bias']:.2f}% exceeds 5%"
        )

    @pytest.mark.slow
    def test_psfex_moffat_stars(self, moffat_field):
        """PSFEx on Moffat stars: bias < 10% (wider tolerance for wing truncation)."""
        image, obj, psfex_dict, fwhm = moffat_field
        stats = _measure_bias(image, obj, psfex_dict, fwhm=fwhm)

        assert stats['n_good'] > 60
        assert abs(stats['med_bias']) < 10.0, (
            f"PSFEx on Moffat: median bias {stats['med_bias']:.2f}% exceeds 10%"
        )

    @pytest.mark.slow
    def test_psfex_better_than_gaussian_on_moffat(self, moffat_field):
        """PSFEx should have smaller bias than Gaussian model on Moffat stars."""
        image, obj, psfex_dict, fwhm = moffat_field

        # PSFEx measurement
        stats_psfex = _measure_bias(image, obj, psfex_dict, fwhm=fwhm)

        # Gaussian model measurement (known mismatch)
        gauss_model = simulation.create_psf_model(
            fwhm, psf_type='gaussian', size=33, oversampling=2,
        )
        stats_gauss = _measure_bias(image, obj, gauss_model, fwhm=fwhm)

        assert stats_psfex['n_good'] > 60 and stats_gauss['n_good'] > 60
        assert abs(stats_psfex['med_bias']) < abs(stats_gauss['med_bias']), (
            f"PSFEx bias {stats_psfex['med_bias']:.2f}% not better than "
            f"Gaussian {stats_gauss['med_bias']:.2f}% on Moffat stars"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Comparison: SEP PSF vs photutils PSF
# ═══════════════════════════════════════════════════════════════════════════

@_skip_no_sep_psf
class TestSEPPSFBiasComparison:
    """Compare SEP PSF photometry with photutils PSF photometry."""

    @pytest.mark.slow
    def test_sep_vs_photutils_gaussian_stars(self):
        """SEP PSF and photutils PSF should both be < 5% bias, differ < 3%."""
        from stdpipe import photometry_psf

        fwhm = 4.0
        n_stars = 50
        fluxes = np.full(n_stars, 10000.0)

        image, obj = _make_star_field(
            n_stars, 256, 40, fwhm, fluxes,
            psf_type='gaussian', noise_std=1.0,
        )

        # SEP PSF measurement
        psf_model = simulation.create_psf_model(
            fwhm, psf_type='gaussian', size=33, oversampling=2,
        )
        stats_sep = _measure_bias(image, obj, psf_model, fwhm=fwhm)

        # photutils PSF measurement
        obj_phot = obj.copy()
        obj_phot['flux'] = fluxes.copy()
        obj_phot['fluxerr'] = np.ones(n_stars)
        result_phot = photometry_psf.measure_objects_psf(
            obj_phot, image, fwhm=fwhm,
            bg=np.full_like(image, 100.0),
            err=np.ones_like(image),
            verbose=False,
        )
        good_phot = result_phot[
            (result_phot['flags'] == 0) & np.isfinite(result_phot['flux'])
        ]
        if len(good_phot) > 0:
            bias_phot = np.median(
                (good_phot['flux'] / good_phot['flux_input'] - 1) * 100
            )
        else:
            bias_phot = np.nan

        # Both should be reasonably accurate
        assert stats_sep['n_good'] > 30
        assert abs(stats_sep['med_bias']) < 5.0, (
            f"SEP PSF bias {stats_sep['med_bias']:.2f}% exceeds 5%"
        )
        if np.isfinite(bias_phot) and len(good_phot) > 30:
            assert abs(bias_phot) < 5.0, (
                f"photutils PSF bias {bias_phot:.2f}% exceeds 5%"
            )
            # Difference between methods
            assert abs(stats_sep['med_bias'] - bias_phot) < 3.0, (
                f"SEP ({stats_sep['med_bias']:.2f}%) vs photutils ({bias_phot:.2f}%): "
                f"differ by > 3%"
            )
