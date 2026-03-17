#!/usr/bin/env python3
"""
Compare PSF photometry methods on a real image with catalog cross-match.

Runs several SEP-based photometry modes on a FITS image and compares
photometric scatter against a reference catalog. The comparison is binned
by nearest-neighbor distance (crowding) and by reference magnitude.

Usage:
    python compare_psf_photometry.py image.fits [--catalog catalog.parquet]
                                                [--mask mask.fits]
                                                [--cat-mag V]
                                                [--mag-range 10 14]
                                                [--fwhm 0]
                                                [--gain 1.0]
                                                [--vignet-size 45]
                                                [--psf-order 2]
                                                [--damp-snthresh 30]

If --catalog is not provided, the script uses Gaia DR3 G magnitudes
(requires WCS in the FITS header and network access).

Requires SEP 1.4+ with psf_fit() support.
"""

import argparse
import sys
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.spatial import cKDTree

from stdpipe import photometry, psf as psf_mod
from stdpipe.photometry_measure import _HAS_SEP_OPTIMAL, measure_objects_sep

if not _HAS_SEP_OPTIMAL:
    print("ERROR: SEP 1.4+ with psf_fit() is required.")
    sys.exit(1)

import sep


# ── helpers ──────────────────────────────────────────────────────────

def mad_sigma(values):
    """MAD-based robust scatter estimate (×1.4826 → Gaussian σ)."""
    med = np.median(values)
    return np.median(np.abs(values - med)) * 1.4826


def load_catalog(args, wcs, image_shape):
    """Load reference catalog and return (x, y, mag) arrays inside the image."""
    ny, nx = image_shape
    margin = 20

    if args.catalog is not None:
        # User-provided catalog (Parquet or FITS table)
        if args.catalog.endswith('.parquet'):
            import pandas as pd
            cat = pd.read_parquet(args.catalog)
            ra, dec = cat['ra'].values, cat['dec'].values
            mag = cat[args.cat_mag].values
        else:
            cat = Table.read(args.catalog)
            ra, dec = np.array(cat['ra']), np.array(cat['dec'])
            mag = np.array(cat[args.cat_mag])

        px, py = wcs.all_world2pix(ra, dec, 0)
    else:
        # Query Gaia DR3 via stdpipe
        from stdpipe import catalogs
        print("Querying Gaia DR3 catalog (requires network)...")
        cat = catalogs.get_cat_vizier(
            wcs=wcs, shape=image_shape,
            catalog='gaiadr3syn', limit=-1,
        )
        if cat is None or len(cat) == 0:
            print("ERROR: No catalog stars found. Provide --catalog explicitly.")
            sys.exit(1)
        px, py = wcs.all_world2pix(cat['ra'], cat['dec'], 0)
        mag = np.array(cat[args.cat_mag])
        print(f"  Got {len(cat)} Gaia stars")

    in_bounds = (
        (px > margin) & (px < nx - margin) &
        (py > margin) & (py < ny - margin) &
        np.isfinite(mag)
    )
    return px[in_bounds], py[in_bounds], mag[in_bounds]


def print_table(title, headers, rows, col_widths=None):
    """Pretty-print a text table."""
    if col_widths is None:
        col_widths = [max(len(h), 8) for h in headers]
    fmt = "  ".join(f"{{:>{w}s}}" for w in col_widths)
    print(f"\n{title}")
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare PSF photometry methods on a real image."
    )
    parser.add_argument("image", help="Input FITS image")
    parser.add_argument("--mask", help="Mask FITS image (True/nonzero = masked)")
    parser.add_argument("--catalog", help="Reference catalog (Parquet or FITS)")
    parser.add_argument(
        "--cat-mag", default="V",
        help="Magnitude column in catalog (default: V)",
    )
    parser.add_argument(
        "--mag-range", nargs=2, type=float, default=[10, 14],
        help="Magnitude range for scatter analysis (default: 10 14)",
    )
    parser.add_argument(
        "--fwhm", type=float, default=0,
        help="Override FWHM in pixels (0 = measure from PSFEx, default: 0)",
    )
    parser.add_argument("--gain", type=float, default=1.0, help="Gain in e-/ADU")
    parser.add_argument(
        "--vignet-size", type=int, default=45,
        help="PSFEx vignet/stamp size (default: 45)",
    )
    parser.add_argument(
        "--psf-order", type=int, default=2,
        help="PSFEx polynomial order (default: 2)",
    )
    parser.add_argument(
        "--damp-snthresh", type=float, default=30.0,
        help="S/N threshold for position damping (default: 30)",
    )
    args = parser.parse_args()

    # ── load data ────────────────────────────────────────────────
    print(f"Loading image: {args.image}")
    image = fits.getdata(args.image).astype(np.float64)
    header = fits.getheader(args.image)
    wcs = WCS(header)

    mask = None
    if args.mask:
        print(f"Loading mask: {args.mask}")
        mask = fits.getdata(args.mask).astype(bool)
    else:
        mask = np.zeros(image.shape, dtype=bool)

    print(f"Image shape: {image.shape}")

    # ── background ───────────────────────────────────────────────
    bkg = sep.Background(image, mask=mask, bw=64, bh=64)
    image_sub = image - bkg.back()
    err = bkg.rms()
    print(f"Background: median={np.median(bkg.back()):.1f}, rms={np.median(err):.2f}")

    # ── PSF model ────────────────────────────────────────────────
    print("Running PSFEx...")
    psfex_model = psf_mod.run_psfex(
        image_sub, mask=mask,
        order=args.psf_order,
        vignet_size=args.vignet_size,
        psf_size=args.vignet_size,
        gain=args.gain,
    )
    fwhm = args.fwhm if args.fwhm > 0 else psfex_model.get('fwhm', 3.0)
    print(f"FWHM: {fwhm:.2f} pix")

    from stdpipe.photometry_measure import _get_sep_psf
    sep_psf = _get_sep_psf(psfex_model, fwhm, print)

    # ── catalog ──────────────────────────────────────────────────
    px, py, vmag = load_catalog(args, wcs, image.shape)
    print(f"Catalog stars in image: {len(px)}")

    # Nearest-neighbor distances
    tree = cKDTree(np.column_stack([px, py]))
    nn_dist, _ = tree.query(np.column_stack([px, py]), k=2)
    nn = nn_dist[:, 1]

    # ── prepare arrays ───────────────────────────────────────────
    image1 = np.ascontiguousarray(image_sub)
    err1 = np.ascontiguousarray(err)
    mask1 = mask.astype(np.uint8)
    aper_pix = 1.5 * fwhm

    obj = Table({'x': px, 'y': py})

    # ── run methods ──────────────────────────────────────────────
    methods = {}

    def run(name, **kwargs):
        t0 = time.time()
        result = measure_objects_sep(obj.copy(), image1, err=err1, gain=args.gain,
                                     mask=mask, bg=np.zeros_like(image1), **kwargs)
        elapsed = time.time() - t0
        flux = np.array(result['flux'], dtype=float)
        fluxerr = np.array(result['fluxerr'], dtype=float)
        print(f"  {name}: {elapsed:.1f}s, {np.sum(np.isfinite(flux) & (flux > 0))}/{len(flux)} valid")
        methods[name] = {'flux': flux, 'fluxerr': fluxerr, 'result': result}

    print(f"\nRunning photometry methods (N={len(px)}, FWHM={fwhm:.2f}, aper={aper_pix:.1f} pix):")

    run("Aperture", aper=1.5, fwhm=fwhm)
    run("Optimal grouped", aper=1.5, fwhm=fwhm, optimal=True, group_sources=True)
    run("PSF ungr fixpos", psf=sep_psf, fwhm=fwhm, group_sources=False, fit_positions=False)
    run("PSF gr fixpos", psf=sep_psf, fwhm=fwhm, group_sources=True, fit_positions=False)
    run("PSF gr fixpos r=2F", psf=sep_psf, fwhm=fwhm, group_sources=True,
        fit_positions=False, fit_radius=2 * fwhm)
    run("PSF ungr fitpos", psf=sep_psf, fwhm=fwhm, group_sources=False, fit_positions=True)
    run("PSF gr fitpos", psf=sep_psf, fwhm=fwhm, group_sources=True, fit_positions=True)
    run(f"PSF gr fitpos r=2F sn={args.damp_snthresh:.0f}", psf=sep_psf, fwhm=fwhm,
        group_sources=True, fit_positions=True, fit_radius=2 * fwhm,
        damp_snthresh=args.damp_snthresh)

    # ── analysis ─────────────────────────────────────────────────
    mag_lo, mag_hi = args.mag_range

    # Compute instrumental magnitudes and zero points
    for name, m in methods.items():
        f = m['flux']
        valid = np.isfinite(f) & (f > 0)
        m['valid'] = valid
        imag = np.full(len(f), np.nan)
        imag[valid] = -2.5 * np.log10(f[valid])
        m['imag'] = imag
        bright = valid & (vmag > mag_lo) & (vmag < mag_lo + 2) & (nn > 10)
        m['zp'] = np.median(vmag[bright] - imag[bright]) if bright.sum() > 10 else np.nan

    # ── summary table ────────────────────────────────────────────
    headers = ["Method", "ZP", "MAD(iso)", "MAD(NN<3)", "MAD(NN<5)", "S/N(bright)"]
    rows = []
    for name, m in methods.items():
        v = m['valid'] & np.isfinite(vmag) & np.isfinite(m['zp'])
        resid = vmag - m['imag'] - m['zp']

        iso = v & (vmag > mag_lo) & (vmag < mag_hi) & (nn > 10)
        close3 = v & (vmag > mag_lo) & (vmag < mag_hi) & (nn < 3)
        close5 = v & (vmag > mag_lo) & (vmag < mag_hi) & (nn < 5)
        sn_sel = v & (vmag > mag_lo + 1) & (vmag < mag_lo + 2) & (nn > 10)

        mad_iso = f"{mad_sigma(resid[iso]):.4f}" if iso.sum() > 5 else "---"
        mad_c3 = f"{mad_sigma(resid[close3]):.4f}" if close3.sum() > 5 else "---"
        mad_c5 = f"{mad_sigma(resid[close5]):.4f}" if close5.sum() > 5 else "---"
        sn = m['flux'][sn_sel] / m['fluxerr'][sn_sel]
        med_sn = f"{np.nanmedian(sn):.1f}" if sn_sel.sum() > 5 else "---"

        rows.append([name, f"{m['zp']:.3f}", mad_iso, mad_c3, mad_c5, med_sn])

    print_table("Photometric Summary", headers, rows, [20, 7, 8, 9, 9, 10])

    # ── scatter by NN distance ───────────────────────────────────
    nn_bins = [(0, 3), (3, 5), (5, 8), (8, 15), (15, 50)]
    headers = ["Method"] + [f"NN{lo}-{hi}" for lo, hi in nn_bins]
    rows = []
    for name, m in methods.items():
        v = m['valid'] & np.isfinite(vmag) & np.isfinite(m['zp'])
        resid = vmag - m['imag'] - m['zp']
        row = [name]
        for lo, hi in nn_bins:
            sel = v & (vmag > mag_lo) & (vmag < mag_hi) & (nn >= lo) & (nn < hi)
            row.append(f"{mad_sigma(resid[sel]):.4f}" if sel.sum() > 5 else "---")
        rows.append(row)

    print_table(
        f"Scatter by NN Distance (mag {mag_lo}-{mag_hi})",
        headers, rows, [20] + [8] * len(nn_bins),
    )

    # ── scatter by magnitude ─────────────────────────────────────
    mag_bins = []
    m0 = mag_lo
    while m0 < mag_hi:
        mag_bins.append((m0, m0 + 1))
        m0 += 1

    headers = ["Method"] + [f"V{lo}-{hi}" for lo, hi in mag_bins]
    rows = []
    for name, m in methods.items():
        v = m['valid'] & np.isfinite(vmag) & np.isfinite(m['zp'])
        resid = vmag - m['imag'] - m['zp']
        row = [name]
        for lo, hi in mag_bins:
            sel = v & (vmag >= lo) & (vmag < hi) & (nn > 10)
            row.append(f"{mad_sigma(resid[sel]):.4f}" if sel.sum() > 5 else "---")
        rows.append(row)

    print_table(
        "Scatter by Magnitude (isolated, NN>10)",
        headers, rows, [20] + [8] * len(mag_bins),
    )

    # ── S/N by magnitude ─────────────────────────────────────────
    headers = ["Method"] + [f"V{lo}-{hi}" for lo, hi in mag_bins]
    rows = []
    for name, m in methods.items():
        v = m['valid']
        with np.errstate(invalid='ignore', divide='ignore'):
            sn = np.where((m['fluxerr'] > 0) & v, m['flux'] / m['fluxerr'], np.nan)
        row = [name]
        for lo, hi in mag_bins:
            sel = v & np.isfinite(vmag) & (vmag >= lo) & (vmag < hi) & (nn > 10)
            row.append(f"{np.nanmedian(sn[sel]):.1f}" if sel.sum() > 5 else "---")
        rows.append(row)

    print_table(
        "Median S/N by Magnitude (isolated, NN>10)",
        headers, rows, [20] + [8] * len(mag_bins),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
