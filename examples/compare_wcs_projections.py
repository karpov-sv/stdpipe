#!/usr/bin/env python3
"""Compare WCS projection fitting for wide-field images.

Detects sources on the image with SEP, matches them to a reference
catalog using ``pipeline.calibrate_photometry()``, then fits TAN-SIP
and ZPN(+SIP) WCS models on the matched centroids. Residual summaries
and diagnostic plots are evaluated on the same cleaned match structure
used by ``plots.plot_photometric_match(..., mode='dist')``.

Usage
-----
Requires a FITS image with an initial WCS and a reference catalog
(Parquet with ra/dec columns):

    python compare_wcs_projections.py image.fits catalog.parquet

The catalog should be a Vizier-style table (e.g. from
stdpipe.catalogs.get_cat_vizier) with 'ra' and 'dec' columns.
"""

import argparse
import time
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table


MATCH_KWARGS = {
    "order": 2,
    "use_color": True,
    "accept_flags": 0x01,
    "max_intrinsic_rms": 0.02,
    "cat_col_ra": "ra",
    "cat_col_dec": "dec",
    "cat_col_mag": "R",
    "cat_col_mag1": "B",
    "cat_col_mag2": "V",
    "update": False,
    "verbose": False,
}


def select_fit_subset_indices(xy, shape, max_points):
    """Down-sample matched detections while preserving radial field coverage."""
    if max_points is None or len(xy) <= max_points:
        return np.arange(len(xy))

    cx, cy = shape[1] / 2, shape[0] / 2
    r = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    order = np.argsort(r)
    return order[np.linspace(0, len(order) - 1, max_points, dtype=int)]


def radial_bins(xy, shape, inner_frac=0.3, outer_frac=0.7):
    """Split stars into center and edge groups using image-scaled radii."""
    cx, cy = shape[1] / 2, shape[0] / 2
    r = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    r_max = max(float(r.max()), 1.0)
    return r <= inner_frac * r_max, r >= outer_frac * r_max


def report(label, dist_arcsec, xy, shape, elapsed):
    """Print a one-line residual summary."""
    center_mask, edge_mask = radial_bins(xy, shape)
    center_med = np.median(dist_arcsec[center_mask]) if np.any(center_mask) else np.nan
    edge_med = np.median(dist_arcsec[edge_mask]) if np.any(edge_mask) else np.nan
    print(
        f"{label:30s}  "
        f"N={len(dist_arcsec):5d}  "
        f"med={np.median(dist_arcsec):6.3f}\"  "
        f"q90={np.percentile(dist_arcsec, 90):6.3f}\"  "
        f"center={center_med:6.3f}\"  "
        f"edge={edge_med:6.3f}\"  "
        f"({elapsed:.1f}s)"
    )


def run_photometric_match(obj, cat, sr_arcsec):
    """Run the cleaned positional+photometric match used for evaluation."""
    from stdpipe import pipeline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pipeline.calibrate_photometry(
            obj,
            cat,
            sr=sr_arcsec / 3600.0,
            **MATCH_KWARGS,
        )


def extract_fit_sample(match, obj, cat, shape, max_matches):
    """Extract matched x/y and sky positions used for WCS fitting."""
    fit_obj_idx = np.asarray(match["oidx"])[match["idx"]]
    fit_cat_idx = np.asarray(match["cidx"])[match["idx"]]

    xy_all = np.column_stack([obj["x"][fit_obj_idx], obj["y"][fit_obj_idx]])
    subset = select_fit_subset_indices(xy_all, shape, max_matches)
    fit_obj_idx = fit_obj_idx[subset]
    fit_cat_idx = fit_cat_idx[subset]

    xy = np.column_stack([obj["x"][fit_obj_idx], obj["y"][fit_obj_idx]])
    sky_sc = SkyCoord(cat["ra"][fit_cat_idx], cat["dec"][fit_cat_idx], unit="deg")
    return xy, sky_sc


def evaluate_wcs_fit(wcs_fit, obj, cat, sr_arcsec):
    """Re-match all detections with the refined WCS for reporting/plots."""
    obj_eval = obj.copy()
    ra_fit, dec_fit = wcs_fit.all_pix2world(obj_eval["x"], obj_eval["y"], 0)
    obj_eval["ra"] = ra_fit
    obj_eval["dec"] = dec_fit
    return run_photometric_match(obj_eval, cat, sr_arcsec)


def get_match_residual_sample(match):
    """Return the cleaned sample used by plot_photometric_match(mode='dist')."""
    if match is None or not np.any(match["idx"]):
        raise RuntimeError("No cleaned matches available for evaluation.")

    dist_arcsec = match["dist"][match["idx"]] * 3600.0
    xy = np.column_stack([match["ox"][match["idx"]], match["oy"][match["idx"]]])
    return dist_arcsec, xy


def require_columns(cat, names):
    missing = [name for name in names if name not in cat.colnames]
    if missing:
        raise ValueError(f"Catalog is missing required columns: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", help="FITS image with initial WCS")
    parser.add_argument("catalog", help="Reference catalog (Parquet)")
    parser.add_argument(
        "--pv-deg", type=int, default=5, help="ZPN PV polynomial degree (default: 5)"
    )
    parser.add_argument(
        "--max-matches",
        "--max-stars",
        dest="max_matches",
        type=int,
        default=5000,
        help="Maximum matched detections to use for fitting (default: 5000, use 0 for all)",
    )
    parser.add_argument(
        "--thresh", type=float, default=5.0, help="SEP detection threshold in sigma (default: 5)"
    )
    parser.add_argument(
        "--sn", type=float, default=5.0, help="Minimum detection S/N (default: 5)"
    )
    parser.add_argument(
        "--match-radius-arcsec",
        type=float,
        default=8.0,
        help="Detection-to-catalog matching radius in arcsec (default: 8)",
    )
    parser.add_argument(
        "--no-centroid",
        action="store_true",
        help="Disable SEP windowed centroid refinement",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plot generation"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Save plot to file instead of showing (e.g. comparison.png)",
    )
    args = parser.parse_args()

    # Load data
    image = fits.getdata(args.image).astype(float)
    header = fits.getheader(args.image)
    shape = image.shape
    wcs_init = WCS(header)
    cat = Table.read(args.catalog)
    require_columns(cat, ["ra", "dec", "R", "B", "V"])

    from stdpipe import photometry
    from stdpipe.astrometry_wcs import (
        fit_wcs_from_points,
        fit_zpn_wcs_from_points,
        tan_wcs_to_zpn,
        _fit_tan_sip_robust,
        _fit_zpn_sip,
    )

    obj = photometry.get_objects_sep(
        image,
        header=header,
        thresh=args.thresh,
        aper=1.0,
        fwhm=True,
        centroid=not args.no_centroid,
        sn=args.sn,
        verbose=False,
    )

    initial_match = run_photometric_match(obj.copy(), cat, args.match_radius_arcsec)
    if initial_match is None or not np.any(initial_match["idx"]):
        raise RuntimeError("Initial detection-to-catalog matching failed.")

    max_matches = None if args.max_matches in (None, 0) else int(args.max_matches)
    xy, sky_sc = extract_fit_sample(initial_match, obj, cat, shape, max_matches=max_matches)

    h, w = shape
    fit_dist_arcsec = initial_match["dist"][initial_match["idx"]] * 3600.0
    print(
        f"Image: {args.image} ({w}x{h}), {len(obj)} detections, "
        f"{np.sum(initial_match['idx'])} clean initial matches"
    )
    print(
        f"Detection: thresh={args.thresh:.1f} sigma, sn={args.sn:.1f}, "
        f"centroid={'off' if args.no_centroid else 'on'}, "
        f"FWHM={obj.meta.get('fwhm_phot', np.nan):.2f} px"
    )
    print(
        f"Initial cleaned match: med={np.median(fit_dist_arcsec):.3f}\", "
        f"q90={np.percentile(fit_dist_arcsec, 90):.3f}\" "
        f"within {args.match_radius_arcsec:.1f}\""
    )
    if max_matches is not None and np.sum(initial_match["idx"]) > len(xy):
        print(
            f"Using a radially stratified fitting subset of {len(xy)} / "
            f"{np.sum(initial_match['idx'])} cleaned matches"
        )
    print()

    # Collect (label, match_result) pairs for plotting
    matches = []

    # --- TAN-SIP at various orders ---
    print("=== TAN-SIP ===")
    for sip_order in [2, 3, 4, 5]:
        try:
            t0 = time.time()
            wcs_tan = _fit_tan_sip_robust(
                (xy[:, 0], xy[:, 1]),
                sky_sc,
                proj_point="center",
                projection=wcs_init,
                sip_degree=sip_order,
            )
            elapsed = time.time() - t0
            m_eval = evaluate_wcs_fit(wcs_tan, obj, cat, args.match_radius_arcsec)
            dist_arcsec, xy_eval = get_match_residual_sample(m_eval)
            label = f"TAN-SIP order={sip_order}"
            report(label, dist_arcsec, xy_eval, shape, elapsed)
            matches.append((label, m_eval))
        except Exception as e:
            print(f"{'TAN-SIP order=' + str(sip_order):30s}  FAILED: {e}")

    # --- ZPN (PV only, no SIP) ---
    print()
    print(f"=== ZPN (pv_deg={args.pv_deg}) ===")
    wcs_zpn = tan_wcs_to_zpn(wcs_init, pv_deg=args.pv_deg)

    t0 = time.time()
    wcs_zpn_only = fit_wcs_from_points(
        xy.T, sky_sc, projection=wcs_zpn, sip_degree=None, pv_deg=args.pv_deg
    )
    elapsed = time.time() - t0
    m_eval = evaluate_wcs_fit(wcs_zpn_only, obj, cat, args.match_radius_arcsec)
    dist_arcsec, xy_eval = get_match_residual_sample(m_eval)
    label = "ZPN PV-only"
    report(label, dist_arcsec, xy_eval, shape, elapsed)
    matches.append((label, m_eval))

    # --- ZPN+SIP at various SIP orders ---
    for sip_order in [2, 3, 4]:
        try:
            t0 = time.time()
            wcs_zpn_sip = fit_wcs_from_points(
                xy.T,
                sky_sc,
                projection=wcs_zpn,
                sip_degree=sip_order,
                pv_deg=args.pv_deg,
            )
            elapsed = time.time() - t0
            m_eval = evaluate_wcs_fit(wcs_zpn_sip, obj, cat, args.match_radius_arcsec)
            dist_arcsec, xy_eval = get_match_residual_sample(m_eval)
            label = f"ZPN+SIP order={sip_order}"
            report(label, dist_arcsec, xy_eval, shape, elapsed)
            matches.append((label, m_eval))

            # Print final PV values
            pv_dict = {}
            for i, m, val in wcs_zpn_sip.wcs.get_pv():
                if i == 2:
                    pv_dict[m] = val
            pvs = ", ".join(f"PV2_{m}={v:.6f}" for m, v in sorted(pv_dict.items()))
            print(f"{'':30s}  PV: {pvs}")
        except Exception as e:
            print(f"{'ZPN+SIP order=' + str(sip_order):30s}  FAILED: {e}")

    # --- Convergence test for best ZPN+SIP ---
    print()
    print("=== ZPN+SIP order=2 convergence ===")
    wcs_zpn_fit, _ = fit_zpn_wcs_from_points(
        xy, sky_sc, wcs_zpn, pv_deg=args.pv_deg
    )
    for n_iter in [1, 2, 3, 5, 8, 10, 15]:
        t0 = time.time()
        wcs_fit = _fit_zpn_sip(
            wcs_zpn_fit, xy, sky_sc, sip_degree=2, pv_deg=args.pv_deg, n_iter=n_iter
        )
        elapsed = time.time() - t0
        m_eval = evaluate_wcs_fit(wcs_fit, obj, cat, args.match_radius_arcsec)
        dist_arcsec, xy_eval = get_match_residual_sample(m_eval)
        report(f"n_iter={n_iter}", dist_arcsec, xy_eval, shape, elapsed)

    # --- Diagnostic plots ---
    if args.no_plot or not matches:
        return

    import matplotlib

    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stdpipe.plots import plot_photometric_match

    n = len(matches)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )

    # Common color scale: use absolute arcsec limits shared across all panels.
    # plot_photometric_match(..., mode='dist') forwards kwargs to binned_map(),
    # where `qq` means percentiles, not absolute values.
    all_dist_arcsec = np.concatenate([m["dist"][m["idx"]] * 3600 for _, m in matches])
    vmin = 0.0
    vmax = np.percentile(all_dist_arcsec, 97.5)

    for i, (label, m) in enumerate(matches):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        plot_photometric_match(m, ax=ax, mode="dist", vmin=vmin, vmax=vmax)
        ax.set_title(label)

    # Hide unused axes
    for i in range(len(matches), nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"\nPlot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
