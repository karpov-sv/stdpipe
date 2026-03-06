#!/usr/bin/env python
"""
Compare SFFT vs HOTPANTS image subtraction on realistic simulated images.

Creates a science image with slightly aberrated (broader) PSF and a sharper
unaberrated template, injects transients with known fluxes, runs both
subtraction methods, and compares:

  - Residual RMS (background noise in difference image)
  - Transient flux recovery accuracy
  - Photometric scatter of non-variable sources
  - Spatial structure in residuals
  - Runtime performance
"""

import numpy as np
import time
import shutil
import warnings

from astropy.stats import mad_std

warnings.filterwarnings('ignore')


def make_test_images(
    size=512,
    n_stars=200,
    template_fwhm=2.5,
    science_fwhm=3.5,
    background=1000.0,
    readnoise=10.0,
    gain=2.0,
    flux_scale=1.08,
    flux_gradient=0.03,
    bg_offset=50.0,
    n_transients=10,
    transient_flux_range=(500, 5000),
    defocus=0.08,
    astigmatism=0.04,
    seed=42,
):
    """Create a realistic science/template image pair.

    Template: sharp PSF (unaberrated), low noise, deep exposure.
    Science:  broader PSF (with optical aberrations), normal noise,
              different flux scale (with spatial gradient), background offset,
              injected transients.
    """
    from stdpipe import simulation, psf as psf_module

    np.random.seed(seed)

    # Common star field
    edge = 30
    star_x = np.random.uniform(edge, size - edge, n_stars)
    star_y = np.random.uniform(edge, size - edge, n_stars)
    star_flux = 10 ** np.random.uniform(2.5, 4.5, n_stars)  # ~300 to ~30000 ADU

    # -- Template: sharp, unaberrated, deep --
    template_psf = simulation.create_psf_model(
        fwhm=template_fwhm, psf_type='moffat', beta=3.0,
        oversampling=2,
    )

    template = np.random.normal(background * 0.5, readnoise * 0.3,
                                 (size, size)).astype(np.float64)
    template += background * 0.5  # lower background for template

    for i in range(n_stars):
        psf_module.place_psf_stamp(template, template_psf,
                                    star_x[i], star_y[i],
                                    flux=star_flux[i])

    # Add Poisson noise
    template_clean = template.copy()
    template = np.random.poisson(
        np.clip(template * gain, 0, None)
    ).astype(np.float64) / gain

    # -- Science: aberrated, noisier, different flux scale --
    science_psf = simulation.create_psf_model(
        fwhm=science_fwhm, psf_type='moffat', beta=3.0,
        oversampling=2,
        defocus=defocus,
        astigmatism_x=astigmatism,
    )

    science = np.random.normal(background, readnoise,
                                (size, size)).astype(np.float64)

    # Spatially varying flux scale
    xn = np.linspace(-1, 1, size)
    yn = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(xn, yn)
    flux_map = flux_scale + flux_gradient * xx

    for i in range(n_stars):
        local_scale = flux_scale + flux_gradient * (star_x[i] - size / 2) / (size / 2)
        psf_module.place_psf_stamp(science, science_psf,
                                    star_x[i], star_y[i],
                                    flux=star_flux[i] * local_scale)

    # Add differential background
    science += bg_offset + 20 * xx + 10 * yy

    # Inject transients (only in science)
    transient_x = np.random.uniform(edge, size - edge, n_transients)
    transient_y = np.random.uniform(edge, size - edge, n_transients)
    transient_flux = np.random.uniform(*transient_flux_range, n_transients)

    for i in range(n_transients):
        psf_module.place_psf_stamp(science, science_psf,
                                    transient_x[i], transient_y[i],
                                    flux=transient_flux[i])

    # Add Poisson noise to science
    science_clean = science.copy()
    science = np.random.poisson(
        np.clip(science * gain, 0, None)
    ).astype(np.float64) / gain

    info = {
        'star_x': star_x,
        'star_y': star_y,
        'star_flux': star_flux,
        'transient_x': transient_x,
        'transient_y': transient_y,
        'transient_flux': transient_flux,
        'template_fwhm': template_fwhm,
        'science_fwhm': science_fwhm,
        'flux_scale': flux_scale,
        'flux_gradient': flux_gradient,
        'bg_offset': bg_offset,
        'flux_map': flux_map,
    }

    return science, template, info


def measure_flux(diff, x, y, aper=8, ann_inner=12, ann_outer=18):
    """Aperture photometry with local background annulus subtraction.

    Measures flux in a circular aperture and subtracts the median
    per-pixel background estimated from a surrounding annulus.
    """
    ny, nx = diff.shape
    yy, xx = np.ogrid[:ny, :nx]
    fluxes = []
    for xi, yi in zip(x, y):
        r2 = (xx - xi)**2 + (yy - yi)**2
        aper_mask = r2 <= aper**2
        ann_mask = (r2 > ann_inner**2) & (r2 <= ann_outer**2)
        n_aper = np.sum(aper_mask)
        # Local background from annulus
        if np.sum(ann_mask) > 10:
            bg_per_pixel = np.median(diff[ann_mask])
        else:
            bg_per_pixel = 0.0
        fluxes.append(np.sum(diff[aper_mask]) - bg_per_pixel * n_aper)
    return np.array(fluxes)


def spatial_rms_map(diff, block=64):
    """Compute block-wise RMS map."""
    ny, nx = diff.shape
    ny_blocks = ny // block
    nx_blocks = nx // block
    rms_map = np.zeros((ny_blocks, nx_blocks))
    for iy in range(ny_blocks):
        for ix in range(nx_blocks):
            tile = diff[iy*block:(iy+1)*block, ix*block:(ix+1)*block]
            rms_map[iy, ix] = mad_std(tile)
    return rms_map


def run_comparison(size=512, verbose=True):
    """Run the full comparison."""

    print("=" * 70)
    print("SFFT vs HOTPANTS Image Subtraction Comparison")
    print("=" * 70)

    # Check HOTPANTS availability
    has_hotpants = shutil.which('hotpants') is not None
    if not has_hotpants:
        print("\nWARNING: HOTPANTS not found. Will run SFFT only.\n")

    # Create test images
    print("\n--- Creating test images ---")
    science, template, info = make_test_images(size=size)
    ny, nx = science.shape
    print(f"  Image size: {nx}x{ny}")
    print(f"  Template FWHM: {info['template_fwhm']:.1f} px")
    print(f"  Science FWHM: {info['science_fwhm']:.1f} px "
          "(aberrated: defocus=0.08, astigmatism=0.04)")
    print(f"  Stars: {len(info['star_x'])}")
    print(f"  Transients: {len(info['transient_x'])}")
    print(f"  Flux scale: {info['flux_scale']:.2f} + {info['flux_gradient']:.2f}*x")
    print(f"  Background offset: {info['bg_offset']:.0f}")

    results = {}

    # ==================== SFFT ====================
    print("\n--- Running SFFT ---")
    from stdpipe.sfft import solve as sfft_solve, evaluate_flux_scale

    sfft_configs = [
        {
            'name': 'SFFT (5x5, poly=1)',
            'kernel_shape': (5, 5),
            'kernel_poly_order': 1,
            'bg_poly_order': 1,
            'flux_poly_order': 1,
        },
        {
            'name': 'SFFT (7x7, poly=2)',
            'kernel_shape': (7, 7),
            'kernel_poly_order': 2,
            'bg_poly_order': 2,
            'flux_poly_order': 1,
        },
        {
            'name': 'SFFT (9x9, poly=2)',
            'kernel_shape': (9, 9),
            'kernel_poly_order': 2,
            'bg_poly_order': 2,
            'flux_poly_order': 1,
        },
        {
            'name': 'SFFT (11x11, poly=2)',
            'kernel_shape': (11, 11),
            'kernel_poly_order': 2,
            'bg_poly_order': 2,
            'flux_poly_order': 2,
        },
    ]

    for cfg in sfft_configs:
        name = cfg.pop('name')
        t0 = time.time()
        r = sfft_solve(science, template, sigma_clip=3.0, max_iter=5,
                     verbose=False, **cfg)
        dt = time.time() - t0

        results[name] = {
            'diff': r.diff,
            'time': dt,
            'rms': r.rms,
            'n_iter': r.n_iter,
            'flux_coeffs': r.flux_poly_coeffs,
        }
        print(f"  {name}: {dt:.2f}s, RMS={r.rms:.2f}, "
              f"iters={r.n_iter}, flux={r.flux_poly_coeffs[0]:.4f}")

    # ==================== HOTPANTS ====================
    if has_hotpants:
        print("\n--- Running HOTPANTS ---")
        from stdpipe.subtraction import run_hotpants

        hotpants_configs = [
            {
                'name': 'HOTPANTS (1x1, ko=0)',
                'nx': 1, 'extra': {'ko': 0, 'bgo': 0},
            },
            {
                'name': 'HOTPANTS (1x1, ko=2)',
                'nx': 1, 'extra': {'ko': 2, 'bgo': 2},
            },
            {
                'name': 'HOTPANTS (2x2, ko=2)',
                'nx': 2, 'extra': {'ko': 2, 'bgo': 2},
            },
        ]

        for cfg in hotpants_configs:
            name = cfg.pop('name')
            t0 = time.time()
            diff = run_hotpants(
                science, template,
                image_fwhm=info['science_fwhm'],
                template_fwhm=info['template_fwhm'],
                err=True, template_err=True,
                image_gain=2.0, template_gain=2.0,
                verbose=False,
                **cfg,
            )
            dt = time.time() - t0

            if diff is None:
                print(f"  {name}: FAILED")
                continue

            rms = mad_std(diff[30:-30, 30:-30])
            results[name] = {
                'diff': diff,
                'time': dt,
                'rms': rms,
            }
            print(f"  {name}: {dt:.2f}s, RMS={rms:.2f}")

    # ==================== Analysis ====================
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    # Header
    print(f"\n{'Method':<30s} {'Time':>6s} {'RMS':>8s} "
          f"{'Med|star|':>10s} {'MAD|star|':>10s} "
          f"{'Trans bias':>11s} {'Trans scatter':>13s}")
    print("-" * 100)

    for name, res in results.items():
        diff = res['diff']

        # Measure residuals at star positions (should be ~0 for non-variable)
        star_resid = measure_flux(
            diff, info['star_x'], info['star_y'], aper=5
        )

        # Measure transient fluxes
        trans_flux = measure_flux(
            diff, info['transient_x'], info['transient_y'], aper=8
        )
        # True transient flux (what should appear in diff)
        true_flux = info['transient_flux']

        flux_ratio = trans_flux / true_flux
        trans_bias = np.median(flux_ratio) - 1.0
        trans_scatter = mad_std(flux_ratio)

        res['star_resid_median'] = np.median(np.abs(star_resid))
        res['star_resid_mad'] = mad_std(star_resid)
        res['trans_bias'] = trans_bias
        res['trans_scatter'] = trans_scatter

        print(f"{name:<30s} {res['time']:>5.1f}s {res['rms']:>8.2f} "
              f"{res['star_resid_median']:>10.1f} {res['star_resid_mad']:>10.1f} "
              f"{trans_bias:>+10.1%} {trans_scatter:>12.1%}")

    # Spatial residual maps
    print("\n--- Spatial RMS maps (block=64) ---")
    for name, res in results.items():
        rms_map = spatial_rms_map(res['diff'], block=64)
        print(f"\n{name}:")
        for row in rms_map:
            print("  " + "  ".join(f"{v:5.1f}" for v in row))

    # ==================== Plots ====================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        method_names = list(results.keys())
        n_methods = len(method_names)

        fig, axes = plt.subplots(3, max(n_methods, 2), figsize=(5 * n_methods, 14))

        for i, name in enumerate(method_names):
            diff = results[name]['diff']

            # Row 1: Difference image (zoomed to center)
            c = size // 2
            hw = min(100, size // 4)
            ax = axes[0, i] if n_methods > 1 else axes[0]
            vmin, vmax = np.percentile(diff[c-hw:c+hw, c-hw:c+hw], [1, 99])
            ax.imshow(diff[c-hw:c+hw, c-hw:c+hw], cmap='RdBu_r',
                      vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title(f'{name}\nRMS={results[name]["rms"]:.1f}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            # Row 2: Histogram of diff pixels
            ax = axes[1, i] if n_methods > 1 else axes[1]
            pixels = diff[30:-30, 30:-30].ravel()
            rms = mad_std(pixels)
            bins = np.linspace(-5 * rms, 5 * rms, 80)
            ax.hist(pixels, bins=bins, density=True, alpha=0.7, color='steelblue')
            # Gaussian reference
            xx = np.linspace(-5 * rms, 5 * rms, 200)
            ax.plot(xx, np.exp(-xx**2 / (2 * rms**2)) / (rms * np.sqrt(2 * np.pi)),
                    'r-', lw=1.5, label=f'Gaussian σ={rms:.1f}')
            ax.legend(fontsize=7)
            ax.set_xlim(-5 * rms, 5 * rms)
            ax.set_title('Pixel distribution', fontsize=9)

            # Row 3: Spatial RMS map
            ax = axes[2, i] if n_methods > 1 else axes[2]
            rms_map = spatial_rms_map(diff, block=64)
            im = ax.imshow(rms_map, cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title('Block RMS', fontsize=9)

        # Hide unused axes
        for i in range(n_methods, axes.shape[1]):
            for row in range(3):
                axes[row, i].set_visible(False)

        plt.suptitle(f'SFFT vs HOTPANTS: {size}x{size}, '
                     f'sci FWHM={info["science_fwhm"]:.1f}, '
                     f'tmpl FWHM={info["template_fwhm"]:.1f}',
                     fontsize=12, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        outname = 'subtraction_comparison.png'
        plt.savefig(outname, dpi=120, bbox_inches='tight')
        print(f"\nPlot saved to {outname}")

    except ImportError:
        print("\nmatplotlib not available, skipping plots")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Compare SFFT and HOTPANTS image subtraction')
    parser.add_argument('--size', type=int, default=512,
                        help='Image size in pixels (default: 512)')
    args = parser.parse_args()

    run_comparison(size=args.size)
