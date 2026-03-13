"""
Synthetic tests exploring photometry limits with Moffat stellar profiles.

Tests background estimation bias, PSF reconstruction accuracy, and photometric
accuracy as functions of:
  - Moffat beta parameter (broad wings: beta=1.5 to near-Gaussian: beta=4.5)
  - Stellar density (sparse to extremely crowded)
  - Photometric method (aperture, optimal extraction, PSF fitting)

The goal is to map out where biases become significant (>1%, >5%, >10%)
for each pipeline stage.

Run with:
  pytest tests/test_moffat_photometry_limits.py -v -s           # all tests
  pytest tests/test_moffat_photometry_limits.py -v -s -m "not slow"  # skip slow ones
"""

import numpy as np
import pytest
from astropy.table import Table

from stdpipe import photometry, photometry_measure, photometry_psf
from stdpipe.photometry_background import get_background
from stdpipe.photometry_measure import _HAS_SEP_OPTIMAL
from stdpipe.simulation import create_psf_model
from stdpipe.psf import place_psf_stamp, create_psf_model as create_epsf_model

if _HAS_SEP_OPTIMAL:
    from stdpipe.photometry_measure import measure_objects_sep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stellar_field(
    size=512,
    nstars=100,
    fwhm=4.0,
    beta=2.5,
    flux_range=(500, 50000),
    bg_level=100.0,
    gain=1.0,
    read_noise=10.0,
    edge_margin=20,
    seed=42,
):
    """
    Generate a synthetic stellar field with Moffat PSF.

    Returns
    -------
    image : ndarray
        Noisy image with stars
    truth : Table
        True positions and fluxes
    psf_model : dict
        PSF model used for injection
    bg_image : ndarray
        Pure background (no stars) for reference
    noiseless : ndarray
        Stars + background without noise
    """
    rng = np.random.RandomState(seed)

    # Create Moffat PSF model
    psf_model = create_psf_model(fwhm=fwhm, psf_type='moffat', beta=beta,
                                 oversampling=4)

    # Random star positions avoiding edges
    x = rng.uniform(edge_margin, size - edge_margin, nstars)
    y = rng.uniform(edge_margin, size - edge_margin, nstars)

    # Power-law flux distribution (more faint stars)
    u = rng.uniform(0, 1, nstars)
    flux = flux_range[0] * (flux_range[1] / flux_range[0]) ** u

    # Build noiseless image
    noiseless = np.full((size, size), bg_level, dtype=np.float64)
    for i in range(nstars):
        place_psf_stamp(noiseless, psf_model, x[i], y[i], flux=flux[i])

    # Add noise: Poisson (signal) + Gaussian (read noise)
    image = noiseless.copy()
    image = rng.poisson(np.maximum(image * gain, 0).astype(np.int64)).astype(
        np.float64
    ) / gain
    image += rng.normal(0, read_noise, image.shape)

    bg_image = np.full((size, size), bg_level, dtype=np.float64)

    truth = Table({
        'x': x,
        'y': y,
        'flux': flux,
        'mag': -2.5 * np.log10(flux),
    })

    return image, truth, psf_model, bg_image, noiseless


def match_catalogs(truth, measured, radius=3.0):
    """
    Cross-match truth and measured catalogs by nearest neighbor.

    Returns indices into truth and measured for matched pairs.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([truth['x'], truth['y']]))
    dists, idx_truth = tree.query(
        np.column_stack([measured['x'], measured['y']]), k=1
    )

    mask = dists < radius
    idx_meas = np.arange(len(measured))[mask]
    idx_truth = idx_truth[mask]

    # Remove duplicates (keep closest match)
    _, unique = np.unique(idx_truth, return_index=True)
    idx_truth = idx_truth[unique]
    idx_meas = idx_meas[unique]

    return idx_truth, idx_meas


def compute_photometry_stats(truth_flux, measured_flux):
    """
    Compute bias and scatter statistics.

    Returns
    -------
    dict with:
        bias_pct : median fractional bias in percent
        scatter_pct : robust scatter (MAD-based) in percent
        outlier_frac : fraction of >10% outliers
    """
    ratio = measured_flux / truth_flux
    bias = np.median(ratio) - 1.0
    mad = np.median(np.abs(ratio - np.median(ratio)))
    scatter = 1.4826 * mad  # MAD to sigma conversion
    outlier_frac = np.mean(np.abs(ratio - 1.0) > 0.10)

    return {
        'bias_pct': bias * 100,
        'scatter_pct': scatter * 100,
        'outlier_frac': outlier_frac,
        'median_ratio': np.median(ratio),
        'n_matched': len(truth_flux),
    }


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# Moffat beta values: 1.5 = very broad wings, 4.5 = nearly Gaussian
BETA_VALUES = [1.5, 2.0, 2.5, 3.5, 4.5]

# Stellar densities (stars per 512x512 image)
DENSITY_CONFIGS = {
    'sparse': 50,
    'moderate': 200,
    'crowded': 800,
    'very_crowded': 2000,
}

FWHM = 4.0
IMAGE_SIZE = 512
BG_LEVEL = 100.0
GAIN = 1.0
READ_NOISE = 10.0


# ---------------------------------------------------------------------------
# Test 1: Background estimation bias vs stellar density and beta
# ---------------------------------------------------------------------------

class TestBackgroundBias:
    """Test how Moffat wings bias background estimation."""

    @pytest.mark.parametrize('beta', BETA_VALUES, ids=[f'beta{b}' for b in BETA_VALUES])
    @pytest.mark.parametrize(
        'density_name,nstars',
        list(DENSITY_CONFIGS.items()),
        ids=list(DENSITY_CONFIGS.keys()),
    )
    def test_sep_background_bias(self, beta, density_name, nstars):
        """SEP background estimator bias from Moffat wing contamination."""
        image, truth, psf_model, bg_image, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        bg_est = get_background(image, method='sep', size=64)

        # Measure bias away from stars (>3*FWHM from any star)
        yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
        star_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        for row in truth:
            r2 = (xx - row['x'])**2 + (yy - row['y'])**2
            star_mask |= r2 < (3 * FWHM)**2

        # Also check global bias (including star regions)
        bg_residual_global = bg_est - BG_LEVEL
        global_bias = np.median(bg_residual_global)

        if (~star_mask).sum() > 100:
            bg_residual_clean = (bg_est - BG_LEVEL)[~star_mask]
            clean_bias = np.median(bg_residual_clean)
        else:
            clean_bias = global_bias

        bias_pct = global_bias / BG_LEVEL * 100

        # Report results
        print(
            f"\n  SEP BG | beta={beta:.1f} density={density_name:>12s} "
            f"({nstars:4d} stars) | "
            f"global_bias={global_bias:+.3f} ({bias_pct:+.2f}%) "
            f"clean_bias={clean_bias:+.3f}"
        )

        # Soft assertion: background should not be biased by more than 20%
        # in any configuration (this catches catastrophic failures)
        assert abs(bias_pct) < 20, (
            f"Catastrophic background bias: {bias_pct:.1f}%"
        )

    @pytest.mark.parametrize('beta', [1.5, 2.5, 4.5], ids=['beta1.5', 'beta2.5', 'beta4.5'])
    @pytest.mark.parametrize(
        'method', ['sep', 'percentile', 'morphology'],
    )
    def test_background_method_comparison(self, beta, method):
        """Compare background methods for crowded Moffat fields."""
        nstars = 500
        image, truth, psf_model, bg_image, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        kwargs = {}
        if method == 'sep':
            kwargs['size'] = 64
        elif method == 'percentile':
            kwargs['size'] = 51
            kwargs['percentile'] = 15.0
        elif method == 'morphology':
            kwargs['size'] = 25

        bg_est = get_background(image, method=method, **kwargs)

        bg_residual = bg_est - BG_LEVEL
        bias = np.median(bg_residual)
        scatter = 1.4826 * np.median(np.abs(bg_residual - bias))
        bias_pct = bias / BG_LEVEL * 100

        print(
            f"\n  BG method={method:>12s} | beta={beta:.1f} "
            f"nstars={nstars} | "
            f"bias={bias:+.3f} ({bias_pct:+.2f}%) "
            f"scatter={scatter:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 2: PSF reconstruction accuracy
# ---------------------------------------------------------------------------

class TestPSFReconstruction:
    """Test ePSF reconstruction fidelity for Moffat profiles."""

    @pytest.mark.slow
    @pytest.mark.parametrize('beta', BETA_VALUES, ids=[f'beta{b}' for b in BETA_VALUES])
    def test_epsf_vs_true_psf(self, beta):
        """Build ePSF from image and compare to true Moffat PSF."""
        # Use moderate density so we have clean, well-separated stars
        nstars = 80
        image, truth, psf_true, bg_image, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            flux_range=(5000, 50000),  # bright stars for PSF building
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        # Detect objects
        obj = photometry.get_objects_sep(
            image, thresh=5.0, aper=FWHM, sn=10.0, verbose=False,
        )

        if len(obj) < 10:
            pytest.skip(f"Too few stars detected ({len(obj)})")

        # Build ePSF from detected stars
        psf_reconstructed = create_epsf_model(
            image, obj=obj, fwhm=FWHM, oversampling=2, verbose=False,
        )

        # Compare PSF stamps at image center
        from stdpipe.psf import get_psf_stamp

        cx, cy = IMAGE_SIZE / 2, IMAGE_SIZE / 2
        stamp_true = get_psf_stamp(psf_true, cx, cy, normalize=True)
        stamp_recon = get_psf_stamp(psf_reconstructed, cx, cy, normalize=True)

        # Ensure same size for comparison
        s = min(stamp_true.shape[0], stamp_recon.shape[0])
        # Center-crop both to same size
        def center_crop(arr, s):
            cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
            hs = s // 2
            return arr[cy - hs:cy - hs + s, cx - hs:cx - hs + s]

        stamp_true = center_crop(stamp_true, s)
        stamp_recon = center_crop(stamp_recon, s)

        # Renormalize after cropping
        stamp_true /= stamp_true.sum()
        stamp_recon /= stamp_recon.sum()

        # Metrics
        residual = stamp_recon - stamp_true
        max_residual = np.max(np.abs(residual))
        rms_residual = np.sqrt(np.mean(residual**2))

        # Enclosed energy comparison at various radii
        yy, xx = np.mgrid[:s, :s]
        rc = (s - 1) / 2.0
        r = np.sqrt((xx - rc)**2 + (yy - rc)**2)

        radii = [1, 2, 3, 5, 8, 12]
        ee_true = []
        ee_recon = []
        for rad in radii:
            mask = r <= rad
            ee_true.append(stamp_true[mask].sum())
            ee_recon.append(stamp_recon[mask].sum())

        ee_true = np.array(ee_true)
        ee_recon = np.array(ee_recon)
        ee_diff = ee_recon - ee_true

        print(
            f"\n  ePSF recon | beta={beta:.1f} | "
            f"max_resid={max_residual:.5f} rms_resid={rms_residual:.6f}"
        )
        for i, rad in enumerate(radii):
            print(
                f"    r<={rad:2d}: true_EE={ee_true[i]:.4f} "
                f"recon_EE={ee_recon[i]:.4f} diff={ee_diff[i]:+.4f} "
                f"({ee_diff[i]/ee_true[i]*100 if ee_true[i]>0 else 0:+.2f}%)"
            )

        # ePSF should capture >90% of the true PSF structure
        assert rms_residual < 0.05, (
            f"ePSF reconstruction RMS too large: {rms_residual:.4f}"
        )

    @pytest.mark.parametrize('beta', [1.5, 2.5, 4.5], ids=['beta1.5', 'beta2.5', 'beta4.5'])
    def test_gaussian_psf_mismatch(self, beta):
        """
        Quantify the error when using a Gaussian PSF model on Moffat data.

        This is the common case: many pipelines assume Gaussian PSF.
        """
        psf_moffat = create_psf_model(fwhm=FWHM, psf_type='moffat', beta=beta,
                                       oversampling=4)
        psf_gaussian = create_psf_model(fwhm=FWHM, psf_type='gaussian',
                                         oversampling=4)

        from stdpipe.psf import get_psf_stamp

        cx, cy = 256.0, 256.0
        stamp_moffat = get_psf_stamp(psf_moffat, cx, cy, normalize=True)
        stamp_gauss = get_psf_stamp(psf_gaussian, cx, cy, normalize=True)

        s = min(stamp_moffat.shape[0], stamp_gauss.shape[0])
        def center_crop(arr, s):
            cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
            hs = s // 2
            return arr[cy - hs:cy - hs + s, cx - hs:cx - hs + s]

        stamp_moffat = center_crop(stamp_moffat, s)
        stamp_gauss = center_crop(stamp_gauss, s)
        stamp_moffat /= stamp_moffat.sum()
        stamp_gauss /= stamp_gauss.sum()

        residual = stamp_gauss - stamp_moffat
        rms = np.sqrt(np.mean(residual**2))

        # Radial profile comparison
        yy, xx = np.mgrid[:s, :s]
        rc = (s - 1) / 2.0
        r = np.sqrt((xx - rc)**2 + (yy - rc)**2)

        print(f"\n  Gaussian vs Moffat(beta={beta:.1f}) mismatch:")
        print(f"    RMS residual = {rms:.6f}")

        # Wing flux comparison
        for rmin, rmax in [(0, 2), (2, 4), (4, 8), (8, 15)]:
            ring = (r >= rmin) & (r < rmax)
            f_moffat = stamp_moffat[ring].sum()
            f_gauss = stamp_gauss[ring].sum()
            if f_moffat > 1e-10:
                diff_pct = (f_gauss - f_moffat) / f_moffat * 100
            else:
                diff_pct = 0
            print(
                f"    r=[{rmin:2d},{rmax:2d}): "
                f"Moffat={f_moffat:.5f} Gauss={f_gauss:.5f} "
                f"diff={diff_pct:+.1f}%"
            )


# ---------------------------------------------------------------------------
# Test 3: Photometric accuracy — aperture, optimal, PSF fitting
# ---------------------------------------------------------------------------

class TestPhotometryAccuracy:
    """
    Compare photometric methods on Moffat fields.

    Measures bias and scatter for:
    - Aperture photometry (various aperture sizes)
    - Optimal extraction (with true Moffat PSF vs wrong Gaussian PSF)
    - PSF fitting photometry (with ePSF)
    """

    @pytest.mark.parametrize('beta', BETA_VALUES, ids=[f'beta{b}' for b in BETA_VALUES])
    @pytest.mark.parametrize(
        'density_name,nstars',
        [('sparse', 50), ('moderate', 200), ('crowded', 800)],
        ids=['sparse', 'moderate', 'crowded'],
    )
    def test_aperture_photometry(self, beta, density_name, nstars):
        """Aperture photometry bias from Moffat wings missing the aperture."""
        image, truth, psf_model, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        # Detect objects
        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=FWHM, sn=5.0, verbose=False,
        )

        if len(obj) < 5:
            pytest.skip(f"Too few detections ({len(obj)})")

        # Test multiple aperture sizes
        apertures = [1.0, 1.5, 2.0, 3.0, 5.0]  # in units of FWHM

        print(f"\n  Aperture phot | beta={beta:.1f} density={density_name}")

        for aper_fwhm in apertures:
            result = photometry_measure.measure_objects(
                obj.copy(), image, aper=aper_fwhm, fwhm=FWHM,
                bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
            )

            # Filter good measurements
            good = result[(result['flags'] == 0) & np.isfinite(result['flux'])]
            if len(good) < 5:
                continue

            idx_truth, idx_meas = match_catalogs(truth, good, radius=FWHM)
            if len(idx_truth) < 5:
                continue

            stats = compute_photometry_stats(
                truth['flux'][idx_truth], good['flux'][idx_meas]
            )

            print(
                f"    aper={aper_fwhm:.1f}×FWHM: "
                f"bias={stats['bias_pct']:+6.2f}% "
                f"scatter={stats['scatter_pct']:5.2f}% "
                f"outliers={stats['outlier_frac']*100:4.1f}% "
                f"(n={stats['n_matched']})"
            )

    @pytest.mark.parametrize('beta', BETA_VALUES, ids=[f'beta{b}' for b in BETA_VALUES])
    @pytest.mark.parametrize(
        'density_name,nstars',
        [('sparse', 50), ('moderate', 200), ('crowded', 800)],
        ids=['sparse', 'moderate', 'crowded'],
    )
    def test_optimal_extraction_true_psf(self, beta, density_name, nstars):
        """Optimal extraction using the true Moffat PSF model."""
        image, truth, psf_model, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=FWHM, sn=5.0, verbose=False,
        )

        if len(obj) < 5:
            pytest.skip(f"Too few detections ({len(obj)})")

        # Optimal extraction with the true Moffat PSF
        result = photometry_measure.measure_objects(
            obj.copy(), image, aper=3.0, fwhm=FWHM,
            psf=psf_model, optimal=True, group_sources=True,
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )

        good = result[(result['flags'] == 0) & np.isfinite(result['flux'])]
        if len(good) < 5:
            pytest.skip("Too few good measurements")

        idx_truth, idx_meas = match_catalogs(truth, good, radius=FWHM)
        if len(idx_truth) < 5:
            pytest.skip("Too few matches")

        stats = compute_photometry_stats(
            truth['flux'][idx_truth], good['flux'][idx_meas]
        )

        print(
            f"\n  Optimal(true PSF) | beta={beta:.1f} "
            f"density={density_name} | "
            f"bias={stats['bias_pct']:+6.2f}% "
            f"scatter={stats['scatter_pct']:5.2f}% "
            f"outliers={stats['outlier_frac']*100:4.1f}% "
            f"(n={stats['n_matched']})"
        )

    @pytest.mark.parametrize('beta', [1.5, 2.0, 2.5, 3.5], ids=[f'beta{b}' for b in [1.5, 2.0, 2.5, 3.5]])
    @pytest.mark.parametrize(
        'density_name,nstars',
        [('sparse', 50), ('moderate', 200), ('crowded', 800)],
        ids=['sparse', 'moderate', 'crowded'],
    )
    def test_optimal_extraction_wrong_gaussian_psf(self, beta, density_name, nstars):
        """
        Optimal extraction using a WRONG Gaussian PSF on Moffat data.

        This simulates the common pipeline assumption of Gaussian PSF.
        """
        image, truth, psf_moffat, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=FWHM, sn=5.0, verbose=False,
        )

        if len(obj) < 5:
            pytest.skip(f"Too few detections ({len(obj)})")

        # Optimal extraction with WRONG Gaussian PSF (same FWHM)
        result = photometry_measure.measure_objects(
            obj.copy(), image, aper=3.0, fwhm=FWHM,
            optimal=True, group_sources=True,  # No psf= → uses Gaussian
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )

        good = result[(result['flags'] == 0) & np.isfinite(result['flux'])]
        if len(good) < 5:
            pytest.skip("Too few good measurements")

        idx_truth, idx_meas = match_catalogs(truth, good, radius=FWHM)
        if len(idx_truth) < 5:
            pytest.skip("Too few matches")

        stats = compute_photometry_stats(
            truth['flux'][idx_truth], good['flux'][idx_meas]
        )

        print(
            f"\n  Optimal(WRONG Gauss PSF) | beta={beta:.1f} "
            f"density={density_name} | "
            f"bias={stats['bias_pct']:+6.2f}% "
            f"scatter={stats['scatter_pct']:5.2f}% "
            f"outliers={stats['outlier_frac']*100:4.1f}% "
            f"(n={stats['n_matched']})"
        )

    @pytest.mark.skipif(not _HAS_SEP_OPTIMAL, reason="Requires SEP 1.4+ with psf_fit")
    @pytest.mark.parametrize('beta', BETA_VALUES, ids=[f'beta{b}' for b in BETA_VALUES])
    @pytest.mark.parametrize(
        'density_name,nstars',
        [('sparse', 50), ('moderate', 200), ('crowded', 500)],
        ids=['sparse', 'moderate', 'crowded'],
    )
    def test_sep_psf_fitting_true_psf(self, beta, density_name, nstars):
        """SEP PSF fitting with true Moffat PSF model."""
        image, truth, psf_model, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=FWHM, sn=5.0, verbose=False,
        )

        if len(obj) < 5:
            pytest.skip(f"Too few detections ({len(obj)})")

        # SEP grouped PSF fitting scales poorly (O(n^2+)), always disable
        result = measure_objects_sep(
            obj.copy(), image, psf=psf_model, fwhm=FWHM,
            bkgann=(3.0, 5.0), group_sources=False,
            gain=GAIN, verbose=False,
        )

        good = result[
            (result['flags'] == 0)
            & np.isfinite(result['flux'])
            & (result['flux'] > 0)
        ]
        if len(good) < 5:
            pytest.skip("Too few good PSF fits")

        idx_truth, idx_meas = match_catalogs(truth, good, radius=FWHM)
        if len(idx_truth) < 5:
            pytest.skip("Too few matches")

        stats = compute_photometry_stats(
            truth['flux'][idx_truth], good['flux'][idx_meas]
        )

        print(
            f"\n  SEP PSF(true Moffat) | beta={beta:.1f} "
            f"density={density_name} | "
            f"bias={stats['bias_pct']:+6.2f}% "
            f"scatter={stats['scatter_pct']:5.2f}% "
            f"outliers={stats['outlier_frac']*100:4.1f}% "
            f"(n={stats['n_matched']})"
        )

    @pytest.mark.skipif(not _HAS_SEP_OPTIMAL, reason="Requires SEP 1.4+ with psf_fit")
    @pytest.mark.parametrize('beta', [1.5, 2.0, 2.5, 3.5], ids=[f'beta{b}' for b in [1.5, 2.0, 2.5, 3.5]])
    @pytest.mark.parametrize(
        'density_name,nstars',
        [('sparse', 50), ('moderate', 200), ('crowded', 500)],
        ids=['sparse', 'moderate', 'crowded'],
    )
    def test_sep_psf_fitting_wrong_gaussian(self, beta, density_name, nstars):
        """SEP PSF fitting with WRONG Gaussian PSF on Moffat data."""
        image, truth, _, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        # Create Gaussian PSF (wrong model for Moffat data)
        psf_gauss = create_psf_model(fwhm=FWHM, psf_type='gaussian',
                                      oversampling=4)

        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=FWHM, sn=5.0, verbose=False,
        )

        if len(obj) < 5:
            pytest.skip(f"Too few detections ({len(obj)})")

        result = measure_objects_sep(
            obj.copy(), image, psf=psf_gauss, fwhm=FWHM,
            bkgann=(3.0, 5.0), group_sources=False,
            gain=GAIN, verbose=False,
        )

        good = result[
            (result['flags'] == 0)
            & np.isfinite(result['flux'])
            & (result['flux'] > 0)
        ]
        if len(good) < 5:
            pytest.skip("Too few good PSF fits")

        idx_truth, idx_meas = match_catalogs(truth, good, radius=FWHM)
        if len(idx_truth) < 5:
            pytest.skip("Too few matches")

        stats = compute_photometry_stats(
            truth['flux'][idx_truth], good['flux'][idx_meas]
        )

        print(
            f"\n  SEP PSF(WRONG Gauss) | beta={beta:.1f} "
            f"density={density_name} | "
            f"bias={stats['bias_pct']:+6.2f}% "
            f"scatter={stats['scatter_pct']:5.2f}% "
            f"outliers={stats['outlier_frac']*100:4.1f}% "
            f"(n={stats['n_matched']})"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize('beta', BETA_VALUES, ids=[f'beta{b}' for b in BETA_VALUES])
    @pytest.mark.parametrize(
        'density_name,nstars',
        [('sparse', 50), ('moderate', 200), ('crowded', 400)],
        ids=['sparse', 'moderate', 'crowded'],
    )
    def test_psf_fitting_photometry(self, beta, density_name, nstars):
        """PSF fitting photometry with ePSF built from the image."""
        image, truth, psf_true, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            flux_range=(1000, 50000),  # brighter for PSF building
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=FWHM, sn=5.0, verbose=False,
        )

        if len(obj) < 10:
            pytest.skip(f"Too few detections ({len(obj)})")

        # Build ePSF from the data itself
        try:
            psf_recon = create_epsf_model(
                image, obj=obj, fwhm=FWHM, oversampling=2, verbose=False,
            )
        except Exception as e:
            pytest.skip(f"ePSF building failed: {e}")

        # PSF fitting photometry
        # Disable grouping for crowded fields (photutils grouped fitting is O(n^2))
        use_grouping = nstars <= 200

        print(
            f"\n  PSF fitting(ePSF) | beta={beta:.1f} "
            f"density={density_name}"
        )

        # --- photutils ---
        result = photometry_psf.measure_objects_psf(
            obj.copy(), image, psf=psf_recon, fwhm=FWHM,
            bkgann=(3.0, 5.0), bkg_order=1,
            group_sources=use_grouping, gain=GAIN, verbose=False,
        )

        good = result[
            (result['flags'] == 0)
            & np.isfinite(result['flux'])
            & (result['flux'] > 0)
        ]
        if len(good) >= 5:
            idx_truth, idx_meas = match_catalogs(truth, good, radius=FWHM)
            if len(idx_truth) >= 5:
                stats = compute_photometry_stats(
                    truth['flux'][idx_truth], good['flux'][idx_meas]
                )
                print(
                    f"    photutils: "
                    f"bias={stats['bias_pct']:+6.2f}% "
                    f"scatter={stats['scatter_pct']:5.2f}% "
                    f"outliers={stats['outlier_frac']*100:4.1f}% "
                    f"(n={stats['n_matched']})"
                )

        # --- SEP PSF fitting ---
        if _HAS_SEP_OPTIMAL:
            result_sep = measure_objects_sep(
                obj.copy(), image, psf=psf_recon, fwhm=FWHM,
                bkgann=(3.0, 5.0), group_sources=False,
                gain=GAIN, verbose=False,
            )

            good_sep = result_sep[
                (result_sep['flags'] == 0)
                & np.isfinite(result_sep['flux'])
                & (result_sep['flux'] > 0)
            ]
            if len(good_sep) >= 5:
                idx_truth, idx_meas = match_catalogs(truth, good_sep, radius=FWHM)
                if len(idx_truth) >= 5:
                    stats = compute_photometry_stats(
                        truth['flux'][idx_truth], good_sep['flux'][idx_meas]
                    )
                    print(
                        f"    SEP:       "
                        f"bias={stats['bias_pct']:+6.2f}% "
                        f"scatter={stats['scatter_pct']:5.2f}% "
                        f"outliers={stats['outlier_frac']*100:4.1f}% "
                        f"(n={stats['n_matched']})"
                    )

    @pytest.mark.parametrize('beta', BETA_VALUES, ids=[f'beta{b}' for b in BETA_VALUES])
    def test_psf_fitting_with_true_psf(self, beta):
        """PSF fitting (photutils + SEP) using the TRUE Moffat PSF model."""
        nstars = 100
        image, truth, psf_true, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=FWHM, sn=5.0, verbose=False,
        )

        if len(obj) < 5:
            pytest.skip(f"Too few detections ({len(obj)})")

        print(f"\n  PSF fitting(TRUE Moffat) | beta={beta:.1f}")

        # --- photutils PSF fitting ---
        result = photometry_psf.measure_objects_psf(
            obj.copy(), image, psf=psf_true, fwhm=FWHM,
            bkgann=(3.0, 5.0), bkg_order=1,
            group_sources=False, gain=GAIN, verbose=False,
        )

        good = result[
            (result['flags'] == 0)
            & np.isfinite(result['flux'])
            & (result['flux'] > 0)
        ]
        if len(good) >= 5:
            idx_truth, idx_meas = match_catalogs(truth, good, radius=FWHM)
            if len(idx_truth) >= 5:
                stats = compute_photometry_stats(
                    truth['flux'][idx_truth], good['flux'][idx_meas]
                )
                print(
                    f"    photutils: "
                    f"bias={stats['bias_pct']:+6.2f}% "
                    f"scatter={stats['scatter_pct']:5.2f}% "
                    f"outliers={stats['outlier_frac']*100:4.1f}% "
                    f"(n={stats['n_matched']})"
                )

        # --- SEP PSF fitting ---
        if _HAS_SEP_OPTIMAL:
            result_sep = measure_objects_sep(
                obj.copy(), image, psf=psf_true, fwhm=FWHM,
                bkgann=(3.0, 5.0), group_sources=False,
                gain=GAIN, verbose=False,
            )

            good_sep = result_sep[
                (result_sep['flags'] == 0)
                & np.isfinite(result_sep['flux'])
                & (result_sep['flux'] > 0)
            ]
            if len(good_sep) >= 5:
                idx_truth, idx_meas = match_catalogs(truth, good_sep, radius=FWHM)
                if len(idx_truth) >= 5:
                    stats = compute_photometry_stats(
                        truth['flux'][idx_truth], good_sep['flux'][idx_meas]
                    )
                    print(
                        f"    SEP:       "
                        f"bias={stats['bias_pct']:+6.2f}% "
                        f"scatter={stats['scatter_pct']:5.2f}% "
                        f"outliers={stats['outlier_frac']*100:4.1f}% "
                        f"(n={stats['n_matched']})"
                    )


# ---------------------------------------------------------------------------
# Test 4: Flux-dependent bias
# ---------------------------------------------------------------------------

class TestFluxDependentBias:
    """Check if photometric bias depends on source brightness."""

    @pytest.mark.parametrize('beta', [1.5, 2.5, 4.5], ids=['beta1.5', 'beta2.5', 'beta4.5'])
    def test_flux_dependent_aperture_bias(self, beta):
        """Aperture photometry bias as a function of source flux."""
        nstars = 300
        image, truth, psf_model, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            flux_range=(100, 100000),  # wide range
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=FWHM, sn=3.0, verbose=False,
        )

        if len(obj) < 10:
            pytest.skip(f"Too few detections ({len(obj)})")

        result = photometry_measure.measure_objects(
            obj.copy(), image, aper=2.0, fwhm=FWHM,
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )

        good = result[(result['flags'] == 0) & np.isfinite(result['flux'])]
        idx_truth, idx_meas = match_catalogs(truth, good, radius=FWHM)

        if len(idx_truth) < 10:
            pytest.skip("Too few matches")

        true_flux = truth['flux'][idx_truth]
        meas_flux = good['flux'][idx_meas]

        # Bin by true flux
        flux_bins = np.logspace(
            np.log10(true_flux.min()), np.log10(true_flux.max()), 6
        )

        print(f"\n  Flux-dependent aperture bias | beta={beta:.1f}")
        for i in range(len(flux_bins) - 1):
            mask = (true_flux >= flux_bins[i]) & (true_flux < flux_bins[i + 1])
            if mask.sum() < 3:
                continue
            ratio = meas_flux[mask] / true_flux[mask]
            bias = (np.median(ratio) - 1) * 100
            scatter = 1.4826 * np.median(np.abs(ratio - np.median(ratio))) * 100
            print(
                f"    flux=[{flux_bins[i]:7.0f}, {flux_bins[i+1]:7.0f}): "
                f"n={mask.sum():3d} bias={bias:+6.2f}% scatter={scatter:5.2f}%"
            )

    @pytest.mark.parametrize('beta', [1.5, 2.5, 4.5], ids=['beta1.5', 'beta2.5', 'beta4.5'])
    def test_flux_dependent_optimal_bias(self, beta):
        """Optimal extraction bias as a function of source flux."""
        nstars = 300
        image, truth, psf_model, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            flux_range=(100, 100000),
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=FWHM, sn=3.0, verbose=False,
        )

        if len(obj) < 10:
            pytest.skip(f"Too few detections ({len(obj)})")

        # Optimal with true PSF
        result_true = photometry_measure.measure_objects(
            obj.copy(), image, aper=3.0, fwhm=FWHM,
            psf=psf_model, optimal=True, group_sources=True,
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )

        # Optimal with wrong Gaussian PSF
        result_gauss = photometry_measure.measure_objects(
            obj.copy(), image, aper=3.0, fwhm=FWHM,
            optimal=True, group_sources=True,
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )

        print(f"\n  Flux-dependent optimal bias | beta={beta:.1f}")

        for label, result in [('true_PSF', result_true), ('gauss_PSF', result_gauss)]:
            good = result[(result['flags'] == 0) & np.isfinite(result['flux'])]
            idx_truth, idx_meas = match_catalogs(truth, good, radius=FWHM)

            if len(idx_truth) < 10:
                continue

            true_flux = truth['flux'][idx_truth]
            meas_flux = good['flux'][idx_meas]

            flux_bins = np.logspace(
                np.log10(true_flux.min()), np.log10(true_flux.max()), 6
            )

            print(f"  [{label}]")
            for i in range(len(flux_bins) - 1):
                mask = (true_flux >= flux_bins[i]) & (true_flux < flux_bins[i + 1])
                if mask.sum() < 3:
                    continue
                ratio = meas_flux[mask] / true_flux[mask]
                bias = (np.median(ratio) - 1) * 100
                scatter = 1.4826 * np.median(np.abs(ratio - np.median(ratio))) * 100
                print(
                    f"    flux=[{flux_bins[i]:7.0f}, {flux_bins[i+1]:7.0f}): "
                    f"n={mask.sum():3d} bias={bias:+6.2f}% scatter={scatter:5.2f}%"
                )


# ---------------------------------------------------------------------------
# Test 5: Very crowded field stress test
# ---------------------------------------------------------------------------

class TestCrowdedFieldLimits:
    """Push crowding to extreme levels to find breakdown points."""

    @pytest.mark.parametrize('beta', [1.5, 2.5, 4.5], ids=['beta1.5', 'beta2.5', 'beta4.5'])
    @pytest.mark.parametrize(
        'nstars', [100, 300, 800, 2000, 4000],
        ids=['n100', 'n300', 'n800', 'n2000', 'n4000'],
    )
    def test_detection_completeness(self, beta, nstars):
        """Detection completeness vs crowding and Moffat beta."""
        image, truth, psf_model, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            flux_range=(500, 50000),
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=FWHM, sn=5.0, verbose=False,
        )

        # Match to truth
        idx_truth, idx_meas = match_catalogs(truth, obj, radius=FWHM)

        completeness = len(idx_truth) / len(truth)

        # Count spurious detections
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([truth['x'], truth['y']]))
        dists, _ = tree.query(np.column_stack([obj['x'], obj['y']]), k=1)
        n_spurious = (dists > FWHM).sum()
        false_positive_rate = n_spurious / len(obj) if len(obj) > 0 else 0

        # Mean nearest-neighbor distance between true stars
        if len(truth) > 1:
            tree2 = cKDTree(np.column_stack([truth['x'], truth['y']]))
            nn_dists, _ = tree2.query(
                np.column_stack([truth['x'], truth['y']]), k=2
            )
            mean_nn = np.median(nn_dists[:, 1])
            crowding_ratio = FWHM / mean_nn  # >1 means overlapping
        else:
            mean_nn = float('inf')
            crowding_ratio = 0

        print(
            f"\n  Completeness | beta={beta:.1f} nstars={nstars:4d} | "
            f"detected={len(obj):4d} matched={len(idx_truth):4d} "
            f"complete={completeness:.1%} "
            f"FP={false_positive_rate:.1%} "
            f"median_NN={mean_nn:.1f}px "
            f"crowd_ratio={crowding_ratio:.2f}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize('beta', [1.5, 2.5], ids=['beta1.5', 'beta2.5'])
    def test_crowded_method_comparison(self, beta):
        """
        Compare all photometric methods in a crowded field.

        This is the key test: which method degrades most gracefully?
        """
        nstars = 600
        image, truth, psf_model, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=FWHM, beta=beta,
            flux_range=(1000, 50000),
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=FWHM, sn=5.0, verbose=False,
        )

        if len(obj) < 10:
            pytest.skip(f"Too few detections ({len(obj)})")

        results = {}

        # 1. Aperture photometry
        r = photometry_measure.measure_objects(
            obj.copy(), image, aper=2.0, fwhm=FWHM,
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )
        results['aperture_2fwhm'] = r

        # 2. Optimal extraction with Gaussian PSF (wrong)
        r = photometry_measure.measure_objects(
            obj.copy(), image, aper=3.0, fwhm=FWHM,
            optimal=True, group_sources=True,
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )
        results['optimal_gauss'] = r

        # 3. Optimal extraction with true Moffat PSF
        r = photometry_measure.measure_objects(
            obj.copy(), image, aper=3.0, fwhm=FWHM,
            psf=psf_model, optimal=True, group_sources=True,
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )
        results['optimal_true'] = r

        # 4. PSF fitting with Gaussian model (no grouping for speed)
        r = photometry_psf.measure_objects_psf(
            obj.copy(), image, fwhm=FWHM,
            bkgann=(3.0, 5.0), bkg_order=1,
            group_sources=False, gain=GAIN, verbose=False,
        )
        results['psf_gauss'] = r

        # 5. PSF fitting with true Moffat PSF (no grouping for speed)
        r = photometry_psf.measure_objects_psf(
            obj.copy(), image, psf=psf_model, fwhm=FWHM,
            bkgann=(3.0, 5.0), bkg_order=1,
            group_sources=False, gain=GAIN, verbose=False,
        )
        results['psf_true'] = r

        # 6-7. SEP photometry (optimal + PSF fitting)
        if _HAS_SEP_OPTIMAL:
            r = measure_objects_sep(
                obj.copy(), image, fwhm=FWHM,
                bkgann=(3.0, 5.0), group_sources=False,
                gain=GAIN, verbose=False,
                optimal=True,
            )
            results['sep_optimal_gauss'] = r

            r = measure_objects_sep(
                obj.copy(), image, psf=psf_model, fwhm=FWHM,
                bkgann=(3.0, 5.0), group_sources=False,
                gain=GAIN, verbose=False,
            )
            results['sep_psf_true'] = r

        print(
            f"\n  Method comparison | beta={beta:.1f} "
            f"nstars={nstars} detected={len(obj)}"
        )

        for method_name, result in results.items():
            good = result[
                (result['flags'] == 0)
                & np.isfinite(result['flux'])
                & (result['flux'] > 0)
            ]
            idx_truth, idx_meas = match_catalogs(truth, good, radius=FWHM)

            if len(idx_truth) < 5:
                print(f"    {method_name:>20s}: too few matches ({len(idx_truth)})")
                continue

            stats = compute_photometry_stats(
                truth['flux'][idx_truth], good['flux'][idx_meas]
            )

            print(
                f"    {method_name:>20s}: "
                f"bias={stats['bias_pct']:+6.2f}% "
                f"scatter={stats['scatter_pct']:5.2f}% "
                f"outliers={stats['outlier_frac']*100:4.1f}% "
                f"(n={stats['n_matched']})"
            )


# ---------------------------------------------------------------------------
# Test 6: FWHM dependence — narrow vs wide PSF
# ---------------------------------------------------------------------------

class TestFWHMDependence:
    """Check how PSF width interacts with Moffat wing effects."""

    @pytest.mark.parametrize('fwhm', [2.0, 3.0, 4.0, 6.0, 8.0],
                             ids=[f'fwhm{f}' for f in [2.0, 3.0, 4.0, 6.0, 8.0]])
    @pytest.mark.parametrize('beta', [1.5, 2.5, 4.5], ids=['beta1.5', 'beta2.5', 'beta4.5'])
    def test_fwhm_vs_beta(self, fwhm, beta):
        """Aperture and optimal photometry bias vs FWHM and beta."""
        nstars = 150
        image, truth, psf_model, _, _ = make_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm=fwhm, beta=beta,
            bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=fwhm, sn=5.0, verbose=False,
        )

        if len(obj) < 5:
            pytest.skip(f"Too few detections ({len(obj)})")

        # Aperture at 2×FWHM
        r_aper = photometry_measure.measure_objects(
            obj.copy(), image, aper=2.0, fwhm=fwhm,
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )

        # Optimal with true PSF
        r_opt = photometry_measure.measure_objects(
            obj.copy(), image, aper=3.0, fwhm=fwhm,
            psf=psf_model, optimal=True, group_sources=True,
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )

        print(f"\n  FWHM={fwhm:.1f} beta={beta:.1f}")

        for label, result in [('aperture', r_aper), ('optimal', r_opt)]:
            good = result[(result['flags'] == 0) & np.isfinite(result['flux'])]
            idx_truth, idx_meas = match_catalogs(truth, good, radius=fwhm)

            if len(idx_truth) < 5:
                print(f"    {label:>10s}: too few matches")
                continue

            stats = compute_photometry_stats(
                truth['flux'][idx_truth], good['flux'][idx_meas]
            )

            print(
                f"    {label:>10s}: "
                f"bias={stats['bias_pct']:+6.2f}% "
                f"scatter={stats['scatter_pct']:5.2f}% "
                f"(n={stats['n_matched']})"
            )


# ---------------------------------------------------------------------------
# Test 7: Spatially varying PSF — ePSF polynomial vs PSFEx
# ---------------------------------------------------------------------------

def make_varying_stellar_field(
    size=512,
    nstars=200,
    fwhm_range=(3.0, 5.0),
    beta=2.5,
    flux_range=(1000, 50000),
    bg_level=100.0,
    gain=1.0,
    read_noise=10.0,
    edge_margin=30,
    seed=42,
):
    """
    Generate a stellar field with spatially varying Moffat PSF.

    FWHM varies linearly from fwhm_range[0] (left edge) to fwhm_range[1]
    (right edge).

    Returns
    -------
    image : ndarray
    truth : Table with 'x', 'y', 'flux', 'fwhm_local'
    noiseless : ndarray
    """
    rng = np.random.RandomState(seed)

    x = rng.uniform(edge_margin, size - edge_margin, nstars)
    y = rng.uniform(edge_margin, size - edge_margin, nstars)

    u = rng.uniform(0, 1, nstars)
    flux = flux_range[0] * (flux_range[1] / flux_range[0]) ** u

    # Position-dependent FWHM: linear gradient in x
    fwhm_local = fwhm_range[0] + (fwhm_range[1] - fwhm_range[0]) * x / size

    noiseless = np.full((size, size), bg_level, dtype=np.float64)
    for i in range(nstars):
        psf_i = create_psf_model(
            fwhm=fwhm_local[i], psf_type='moffat', beta=beta, oversampling=4
        )
        place_psf_stamp(noiseless, psf_i, x[i], y[i], flux=flux[i])

    image = noiseless.copy()
    image = rng.poisson(
        np.maximum(image * gain, 0).astype(np.int64)
    ).astype(np.float64) / gain
    image += rng.normal(0, read_noise, image.shape)

    truth = Table({
        'x': x,
        'y': y,
        'flux': flux,
        'fwhm_local': fwhm_local,
    })

    return image, truth, noiseless


class TestSpatiallyVaryingPSF:
    """
    Compare ePSF polynomial (degree=1) vs PSFEx (order=1) on a field with
    spatially varying Moffat PSF (FWHM gradient), measuring both accuracy
    and performance.

    All PSF photometry uses background-subtracted images for reliable
    convergence. Flag filtering excludes PSF fit failures (0x1000) but
    allows centroid shifts (0x2000) which are common with imperfect PSF models.
    """

    FWHM_RANGE = (3.0, 5.0)
    MEAN_FWHM = 4.0
    # Mask for PSF-specific failure flags only (0x1000 = fit failed)
    PSF_FAIL_MASK = 0x1000

    def _prepare_field(self, nstars, beta):
        """Create field, detect objects, background-subtract."""
        image, truth, _ = make_varying_stellar_field(
            size=IMAGE_SIZE, nstars=nstars, fwhm_range=self.FWHM_RANGE,
            beta=beta, bg_level=BG_LEVEL, gain=GAIN, read_noise=READ_NOISE,
        )

        # Detect on raw image (better S/N for detection)
        obj = photometry.get_objects_sep(
            image, thresh=3.0, aper=self.MEAN_FWHM, sn=5.0, verbose=False,
        )

        # Background-subtract for PSF building and photometry
        bg = get_background(image, method='sep', size=64)
        image_bgsub = image - bg

        return image, image_bgsub, truth, obj

    def _filter_good(self, result):
        """Filter to objects with successful PSF fits."""
        return result[
            (result['flags'] & self.PSF_FAIL_MASK == 0)
            & np.isfinite(result['flux'])
            & (result['flux'] > 0)
        ]

    def _report_stats(self, label, truth, good, t_psf=None, t_phot=None):
        """Match to truth and print photometry stats."""
        timing = ""
        if t_psf is not None:
            timing += f"PSF: {t_psf:.2f}s  "
        if t_phot is not None:
            timing += f"Phot: {t_phot:.2f}s"

        if len(good) < 5:
            print(f"    {label}: too few good ({len(good)})  {timing}")
            return None

        idx_truth, idx_meas = match_catalogs(truth, good, radius=self.MEAN_FWHM)
        if len(idx_truth) < 5:
            print(f"    {label}: too few matches  {timing}")
            return None

        stats = compute_photometry_stats(
            truth['flux'][idx_truth], good['flux'][idx_meas]
        )

        # Position-dependent bias (left vs right half)
        t_matched = truth[idx_truth]
        g_matched = good[idx_meas]
        left = t_matched['x'] < IMAGE_SIZE / 2
        right = ~left
        bias_left = bias_right = float('nan')
        if left.sum() >= 3:
            ratio_l = g_matched['flux'][left] / t_matched['flux'][left]
            bias_left = (np.median(ratio_l) - 1) * 100
        if right.sum() >= 3:
            ratio_r = g_matched['flux'][right] / t_matched['flux'][right]
            bias_right = (np.median(ratio_r) - 1) * 100

        print(
            f"    {label}: "
            f"bias={stats['bias_pct']:+6.2f}% "
            f"scatter={stats['scatter_pct']:5.2f}% "
            f"outliers={stats['outlier_frac']*100:4.1f}% "
            f"n={stats['n_matched']} "
            f"L={bias_left:+.1f}% R={bias_right:+.1f}%"
            f"  {timing}"
        )
        return stats

    @pytest.mark.slow
    @pytest.mark.parametrize(
        'beta', [1.5, 2.5, 4.5], ids=['beta1.5', 'beta2.5', 'beta4.5']
    )
    @pytest.mark.parametrize(
        'nstars', [100, 300, 800], ids=['n100', 'n300', 'n800']
    )
    def test_epsf_poly_accuracy(self, beta, nstars):
        """ePSF polynomial (degree=1) photometry on spatially varying field."""
        import time

        image, image_bgsub, truth, obj = self._prepare_field(nstars, beta)
        if len(obj) < 10:
            pytest.skip(f"Too few detections ({len(obj)})")

        t0 = time.time()
        try:
            psf_epsf = create_epsf_model(
                image_bgsub, obj=obj, fwhm=self.MEAN_FWHM,
                oversampling=2, degree=1, verbose=False,
            )
        except Exception as e:
            pytest.skip(f"ePSF degree=1 build failed: {e}")
        t_psf = time.time() - t0

        t0 = time.time()
        result = photometry_psf.measure_objects_psf(
            obj.copy(), image_bgsub, psf=psf_epsf, fwhm=self.MEAN_FWHM,
            use_position_dependent_psf=True,
            group_sources=False,
            gain=GAIN, verbose=False,
        )
        t_phot = time.time() - t0

        print(f"\n  ePSF poly d=1 | beta={beta:.1f} nstars={nstars}")
        good = self._filter_good(result)
        self._report_stats('ePSF_d1', truth, good, t_psf, t_phot)

    @pytest.mark.slow
    @pytest.mark.requires_sextractor
    @pytest.mark.requires_psfex
    @pytest.mark.parametrize(
        'beta', [1.5, 2.5, 4.5], ids=['beta1.5', 'beta2.5', 'beta4.5']
    )
    @pytest.mark.parametrize(
        'nstars', [100, 300, 800], ids=['n100', 'n300', 'n800']
    )
    def test_psfex_accuracy(self, beta, nstars):
        """PSFEx (order=1) photometry on spatially varying field."""
        import time
        from stdpipe.psf import run_psfex

        image, image_bgsub, truth, obj = self._prepare_field(nstars, beta)
        if len(obj) < 10:
            pytest.skip(f"Too few detections ({len(obj)})")

        t0 = time.time()
        vignet_size = max(25, int(np.ceil(8 * self.MEAN_FWHM)))
        if vignet_size % 2 == 0:
            vignet_size += 1
        psf_psfex = run_psfex(
            image, order=1, gain=GAIN,
            vignet_size=vignet_size,
            verbose=False,
        )
        t_psf = time.time() - t0

        if psf_psfex is None:
            pytest.skip("PSFEx binary not found")

        t0 = time.time()
        result = photometry_psf.measure_objects_psf(
            obj.copy(), image_bgsub, psf=psf_psfex, fwhm=self.MEAN_FWHM,
            use_position_dependent_psf=True,
            group_sources=False,
            gain=GAIN, verbose=False,
        )
        t_phot = time.time() - t0

        print(f"\n  PSFEx order=1 | beta={beta:.1f} nstars={nstars}")
        good = self._filter_good(result)
        self._report_stats('PSFEx', truth, good, t_psf, t_phot)

    @pytest.mark.slow
    @pytest.mark.requires_sextractor
    @pytest.mark.requires_psfex
    @pytest.mark.parametrize(
        'beta', [1.5, 2.5, 4.5], ids=['beta1.5', 'beta2.5', 'beta4.5']
    )
    def test_epsf_vs_psfex_comparison(self, beta):
        """
        Head-to-head comparison of ePSF polynomial vs PSFEx on the same
        spatially varying field. Reports PSF shape accuracy, photometric
        accuracy, and timing.
        """
        import time
        from stdpipe.psf import run_psfex, get_psf_stamp

        nstars = 300
        image, image_bgsub, truth, obj = self._prepare_field(nstars, beta)
        if len(obj) < 10:
            pytest.skip(f"Too few detections ({len(obj)})")

        # --- Build both PSF models ---
        vignet_size = max(25, int(np.ceil(8 * self.MEAN_FWHM)))
        if vignet_size % 2 == 0:
            vignet_size += 1

        t0 = time.time()
        try:
            psf_epsf = create_epsf_model(
                image_bgsub, obj=obj, fwhm=self.MEAN_FWHM,
                oversampling=2, degree=1, verbose=False,
            )
        except Exception as e:
            pytest.skip(f"ePSF build failed: {e}")
        t_epsf_build = time.time() - t0

        t0 = time.time()
        psf_psfex = run_psfex(
            image, order=1, gain=GAIN,
            vignet_size=vignet_size,
            verbose=False,
        )
        t_psfex_build = time.time() - t0

        if psf_psfex is None:
            pytest.skip("PSFEx binary not found")

        # --- Compare PSF stamps at different positions ---
        print(f"\n  ePSF vs PSFEx head-to-head | beta={beta:.1f} nstars={nstars}")
        print(f"  Build time: ePSF={t_epsf_build:.2f}s  PSFEx={t_psfex_build:.2f}s")

        positions = [
            (IMAGE_SIZE * 0.15, IMAGE_SIZE / 2),
            (IMAGE_SIZE / 2, IMAGE_SIZE / 2),
            (IMAGE_SIZE * 0.85, IMAGE_SIZE / 2),
        ]
        for px, py in positions:
            fwhm_here = self.FWHM_RANGE[0] + (
                self.FWHM_RANGE[1] - self.FWHM_RANGE[0]
            ) * px / IMAGE_SIZE
            psf_true_here = create_psf_model(
                fwhm=fwhm_here, psf_type='moffat', beta=beta, oversampling=4
            )
            stamp_true = get_psf_stamp(psf_true_here, px, py, normalize=True)
            stamp_epsf = get_psf_stamp(psf_epsf, px, py, normalize=True)
            stamp_psfex = get_psf_stamp(psf_psfex, px, py, normalize=True)

            s = min(stamp_true.shape[0], stamp_epsf.shape[0], stamp_psfex.shape[0])

            def center_crop(arr, s):
                cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
                hs = s // 2
                return arr[cy - hs:cy - hs + s, cx - hs:cx - hs + s]

            st = center_crop(stamp_true, s)
            se = center_crop(stamp_epsf, s)
            sp = center_crop(stamp_psfex, s)
            st /= st.sum()
            se /= se.sum()
            sp /= sp.sum()

            rms_epsf = np.sqrt(np.mean((se - st)**2))
            rms_psfex = np.sqrt(np.mean((sp - st)**2))
            peak_epsf = se.max() / st.max()
            peak_psfex = sp.max() / st.max()

            print(
                f"  x={px:5.0f} (FWHM={fwhm_here:.2f}): "
                f"ePSF rms={rms_epsf:.5f} peak={peak_epsf:.4f}  "
                f"PSFEx rms={rms_psfex:.5f} peak={peak_psfex:.4f}"
            )

        # --- Photometry comparison ---
        print("  Photometry:")
        models = {
            'ePSF_poly1': (psf_epsf, True, t_epsf_build),
            'PSFEx_ord1': (psf_psfex, True, t_psfex_build),
        }

        # Also test constant PSF (degree=0 ePSF) as baseline
        try:
            psf_const = create_epsf_model(
                image_bgsub, obj=obj, fwhm=self.MEAN_FWHM,
                oversampling=2, degree=0, verbose=False,
            )
            models['ePSF_const'] = (psf_const, False, None)
        except Exception:
            pass

        for name, (psf_model, use_pos_dep, t_build) in models.items():
            t0 = time.time()
            r = photometry_psf.measure_objects_psf(
                obj.copy(), image_bgsub, psf=psf_model, fwhm=self.MEAN_FWHM,
                use_position_dependent_psf=use_pos_dep,
                group_sources=False, gain=GAIN, verbose=False,
            )
            t_phot = time.time() - t0
            good = self._filter_good(r)
            self._report_stats(f'{name:>12s}', truth, good, t_build, t_phot)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        'beta', [1.5, 2.5, 4.5], ids=['beta1.5', 'beta2.5', 'beta4.5']
    )
    def test_constant_vs_varying_psf_photometry(self, beta):
        """
        Quantify the photometric penalty of using a constant PSF model
        on a field with spatially varying PSF.
        """
        nstars = 200
        image, image_bgsub, truth, obj = self._prepare_field(nstars, beta)
        if len(obj) < 10:
            pytest.skip(f"Too few detections ({len(obj)})")

        # Constant ePSF (degree=0)
        try:
            psf_const = create_epsf_model(
                image_bgsub, obj=obj, fwhm=self.MEAN_FWHM,
                oversampling=2, degree=0, verbose=False,
            )
        except Exception as e:
            pytest.skip(f"ePSF degree=0 failed: {e}")

        # Varying ePSF (degree=1)
        try:
            psf_vary = create_epsf_model(
                image_bgsub, obj=obj, fwhm=self.MEAN_FWHM,
                oversampling=2, degree=1, verbose=False,
            )
        except Exception as e:
            pytest.skip(f"ePSF degree=1 failed: {e}")

        print(f"\n  Constant vs varying PSF penalty | beta={beta:.1f}")

        methods = {}

        # PSF photometry: constant
        r = photometry_psf.measure_objects_psf(
            obj.copy(), image_bgsub, psf=psf_const, fwhm=self.MEAN_FWHM,
            group_sources=False, gain=GAIN, verbose=False,
        )
        methods['PSF_const_d0'] = r

        # PSF photometry: varying
        r = photometry_psf.measure_objects_psf(
            obj.copy(), image_bgsub, psf=psf_vary, fwhm=self.MEAN_FWHM,
            use_position_dependent_psf=True,
            group_sources=False, gain=GAIN, verbose=False,
        )
        methods['PSF_vary_d1'] = r

        # Aperture photometry as baseline
        r = photometry_measure.measure_objects(
            obj.copy(), image, aper=2.0, fwhm=self.MEAN_FWHM,
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )
        methods['aperture_2x'] = r

        # Optimal with Gaussian (no spatial dependence)
        r = photometry_measure.measure_objects(
            obj.copy(), image, aper=3.0, fwhm=self.MEAN_FWHM,
            optimal=True, group_sources=False,
            bkgann=(3.0, 5.0), gain=GAIN, verbose=False,
        )
        methods['optimal_gauss'] = r

        for name, result in methods.items():
            good = self._filter_good(result)
            self._report_stats(f'{name:>15s}', truth, good)
