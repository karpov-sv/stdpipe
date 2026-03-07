#!/usr/bin/env python
"""
Standalone testing and benchmarking script for flux conservation
in templates.reproject_swarp() vs reproject package routines.

Tests performed:
1. Total flux conservation under reprojection (shift, rotation, scale change)
2. Point source flux conservation (individual stars)
3. Surface brightness conservation (uniform and gradient fields)
4. Pixel area correction (when pixel scale changes)
5. Timing benchmarks

Requires: stdpipe, reproject, astropy, numpy, matplotlib

Note on SWarp: SWarp computes internal weights from background RMS. Images
with zero background produce zero weights and all-NaN output. We add small
Gaussian noise for SWarp to get valid weights without biasing flux.

Note on flux vs surface brightness conservation:
- "Flux-conserving" methods (SWarp, reproject_adaptive conserve_flux=True)
  preserve total flux: when pixels get larger, pixel *values* increase by
  the area ratio so that sum(pixels) stays constant.
- "Surface-brightness-conserving" methods (reproject_interp, reproject_exact,
  reproject_adaptive conserve_flux=False) preserve per-pixel values: when
  pixels get larger, pixel values stay the same but total flux (sum) decreases
  by 1/area_ratio.

Both behaviors are correct for their intended use case. This script measures
whether each method is self-consistent: after correcting for the known
area ratio factor, the residual error should be ~0.
"""

import time
import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from collections import OrderedDict

# reproject methods
from reproject import reproject_interp, reproject_adaptive, reproject_exact

# stdpipe
from stdpipe.templates import reproject_swarp
from stdpipe.reproject import reproject_lanczos as stdpipe_reproject_lanczos

from scipy.ndimage import map_coordinates as scipy_map_coordinates
from astropy.wcs.utils import pixel_to_pixel


# ---------------------------------------------------------------------------
# Lanczos interpolation
# ---------------------------------------------------------------------------

def lanczos_kernel(x, a=3):
    """Lanczos kernel of order a."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    # sinc(x) * sinc(x/a) for |x| < a, 0 otherwise
    mask = np.abs(x) < a
    # Handle x=0 separately
    zero = x == 0
    result[zero] = 1.0
    nonzero = mask & ~zero
    xn = x[nonzero]
    result[nonzero] = (np.sin(np.pi * xn) * np.sin(np.pi * xn / a)
                        / (np.pi * xn * np.pi * xn / a))
    return result


def lanczos_map_coordinates(image, coords, a=3, cval=np.nan):
    """Interpolate image at fractional pixel coordinates using Lanczos kernel.

    Parameters
    ----------
    image : 2D array
    coords : (2, N) array of (row, col) coordinates
    a : int, Lanczos kernel order (2, 3, or 4 typical)
    cval : fill value for out-of-bounds

    Returns
    -------
    values : 1D array of interpolated values
    """
    ny, nx = image.shape
    n_pts = coords.shape[1]
    result = np.full(n_pts, cval, dtype=np.float64)

    yr, xr = coords[0], coords[1]

    # Filter out-of-bounds
    valid = (yr >= -0.5) & (yr < ny - 0.5) & (xr >= -0.5) & (xr < nx - 0.5)

    yr_v = yr[valid]
    xr_v = xr[valid]

    # Integer and fractional parts
    iy = np.floor(yr_v).astype(int)
    ix = np.floor(xr_v).astype(int)
    fy = yr_v - iy
    fx = xr_v - ix

    # Kernel support: -a+1 to a
    offsets = np.arange(-a + 1, a + 1)  # 2a values

    # Precompute kernels for all points: shape (n_valid, 2*a)
    ky = lanczos_kernel(fy[:, None] - offsets[None, :], a)  # (n_valid, 2a)
    kx = lanczos_kernel(fx[:, None] - offsets[None, :], a)

    # Normalize kernels
    ky /= ky.sum(axis=1, keepdims=True)
    kx /= kx.sum(axis=1, keepdims=True)

    # Row and column indices: (n_valid, 2a)
    row_idx = iy[:, None] + offsets[None, :]
    col_idx = ix[:, None] + offsets[None, :]

    # Clip to image boundaries (reflect at edges)
    row_idx = np.clip(row_idx, 0, ny - 1)
    col_idx = np.clip(col_idx, 0, nx - 1)

    # Gather image values and apply separable kernel
    # For each valid point, sum over 2a x 2a neighborhood
    vals = np.zeros(len(yr_v))
    for j, oj in enumerate(offsets):
        row_j = row_idx[:, j]
        kj = ky[:, j]
        for k, ok in enumerate(offsets):
            col_k = col_idx[:, k]
            vals += kj * kx[:, k] * image[row_j, col_k]

    result[valid] = vals
    return result


def reproject_interp_custom(input_data, output_header, shape_out=None,
                            order=1, kernel='scipy'):
    """Minimal reproject_interp clone supporting scipy splines and Lanczos.

    Parameters
    ----------
    input_data : tuple of (array, header/WCS)
    output_header : FITS header or WCS
    shape_out : tuple
    order : int, interpolation order (1=bilinear, 3=bicubic for scipy;
            or Lanczos kernel size if kernel='lanczos')
    kernel : 'scipy' or 'lanczos'

    Returns
    -------
    result, footprint
    """
    image, hdr_in = input_data
    wcs_in = WCS(hdr_in)
    wcs_out = WCS(output_header)

    if shape_out is None:
        shape_out = (output_header['NAXIS2'], output_header['NAXIS1'])

    ny_out, nx_out = shape_out

    # Output pixel grid
    yy, xx = np.mgrid[0:ny_out, 0:nx_out]
    pixel_out_x = xx.ravel().astype(float)
    pixel_out_y = yy.ravel().astype(float)

    # Transform output pixels -> input pixels
    pixel_in = pixel_to_pixel(wcs_out, wcs_in, pixel_out_x, pixel_out_y)
    pixel_in_x = np.asarray(pixel_in[0])
    pixel_in_y = np.asarray(pixel_in[1])

    # Interpolate
    coords = np.array([pixel_in_y, pixel_in_x])

    if kernel == 'lanczos':
        values = lanczos_map_coordinates(image.astype(np.float64), coords, a=order)
    else:
        # scipy spline
        values = scipy_map_coordinates(
            image.astype(np.float32), coords,
            order=order, mode='constant', cval=np.nan
        )

    result = values.reshape(shape_out)

    # Footprint: 1 where valid, 0 where NaN
    footprint = (~np.isnan(result)).astype(float)
    return result, footprint


def reproject_interp_oversampled(input_data, output_header, shape_out=None,
                                  order=3, kernel='lanczos', oversamp=None):
    """Lanczos reproject with SWarp-style oversampling and Jacobian scaling.

    When output pixels are larger than input pixels, evaluates the interpolation
    at multiple sub-pixel positions within each output pixel and averages,
    then multiplies by the pixel area ratio (Jacobian) for flux conservation.

    Parameters
    ----------
    input_data : tuple of (array, header/WCS)
    output_header : FITS header or WCS
    shape_out : tuple
    order : int, Lanczos kernel order
    kernel : 'lanczos' or 'scipy'
    oversamp : int or None
        Oversampling factor per axis. None = automatic (like SWarp).
    """
    image, hdr_in = input_data
    wcs_in = WCS(hdr_in)
    wcs_out = WCS(output_header)

    if shape_out is None:
        shape_out = (output_header['NAXIS2'], output_header['NAXIS1'])

    ny_out, nx_out = shape_out

    # Determine pixel scale ratio for automatic oversampling
    from astropy.wcs.utils import proj_plane_pixel_scales
    scale_in = np.mean(proj_plane_pixel_scales(wcs_in))
    scale_out = np.mean(proj_plane_pixel_scales(wcs_out))
    scale_ratio = scale_out / scale_in  # >1 means output pixels larger

    if oversamp is None:
        # SWarp-style: oversample when downsampling (output pixels larger)
        oversamp = max(1, int(scale_ratio + 0.5))

    # Area ratio for Jacobian scaling (flux conservation)
    area_ratio = (scale_out / scale_in) ** 2

    if oversamp <= 1:
        # No oversampling needed — just do Lanczos + Jacobian
        result, footprint = reproject_interp_custom(
            input_data, output_header, shape_out=shape_out,
            order=order, kernel=kernel
        )
        result *= area_ratio
        return result, footprint

    # Oversampled: evaluate at sub-pixel grid within each output pixel
    step = 1.0 / oversamp
    offsets = np.arange(oversamp) * step + step / 2 - 0.5  # centered in pixel

    accumulator = np.zeros(shape_out, dtype=np.float64)
    count = np.zeros(shape_out, dtype=np.int32)

    for dy_off in offsets:
        for dx_off in offsets:
            # Sub-pixel output positions
            yy, xx = np.mgrid[0:ny_out, 0:nx_out]
            pixel_out_x = (xx + dx_off).ravel().astype(float)
            pixel_out_y = (yy + dy_off).ravel().astype(float)

            # Transform to input pixels
            pixel_in = pixel_to_pixel(wcs_out, wcs_in, pixel_out_x, pixel_out_y)
            pixel_in_x = np.asarray(pixel_in[0])
            pixel_in_y = np.asarray(pixel_in[1])

            coords = np.array([pixel_in_y, pixel_in_x])

            if kernel == 'lanczos':
                values = lanczos_map_coordinates(
                    image.astype(np.float64), coords, a=order
                )
            else:
                values = scipy_map_coordinates(
                    image.astype(np.float32), coords,
                    order=order, mode='constant', cval=np.nan
                )

            vals_2d = values.reshape(shape_out)
            valid = np.isfinite(vals_2d)
            accumulator[valid] += vals_2d[valid]
            count[valid] += 1

    # Average over sub-samples, then apply Jacobian area ratio
    result = np.full(shape_out, np.nan, dtype=np.float64)
    good = count > 0
    result[good] = (accumulator[good] / count[good]) * area_ratio

    footprint = good.astype(float)
    return result, footprint


# ---------------------------------------------------------------------------
# Which methods conserve flux vs surface brightness
# ---------------------------------------------------------------------------

# For SB-conserving methods under pixel scale change S, the expected
# star flux ratio is 1/S^2 (fewer pixels, same pixel value).
# For flux-conserving methods, it is 1.0.
FLUX_CONSERVING = {
    'SWarp': True,
    'interp\n(bilinear)': False,
    'interp\n(bicubic)': False,
    'Lanczos-3': False,
    'Lanczos-3\n+oversamp': True,
    'Lanczos-3\n+oversamp4': True,
    'adaptive\n(conserve)': True,
    'adaptive\n(no conserve)': False,
    'exact': False,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_wcs(crpix, crval, cdelt, shape, ctype=('RA---TAN', 'DEC--TAN')):
    """Create a simple TAN WCS."""
    w = WCS(naxis=2)
    w.wcs.crpix = crpix
    w.wcs.crval = crval
    w.wcs.cdelt = cdelt
    w.wcs.ctype = list(ctype)
    w.pixel_shape = shape[::-1]  # (nx, ny)
    return w


def make_header(wcs, shape):
    """Make a FITS header from WCS + shape."""
    hdr = wcs.to_header(relax=True)
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = shape[1]
    hdr['NAXIS2'] = shape[0]
    hdr['BITPIX'] = -64
    return hdr


def gaussian2d(x, y, x0, y0, sigma, flux):
    """Normalized 2D Gaussian with given total flux."""
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return flux / (2 * np.pi * sigma ** 2) * np.exp(-r2 / (2 * sigma ** 2))


def make_star_field(shape, nstars=5, sigma=3.0, flux=10000.0, seed=42):
    """Create a zero-background image with well-separated Gaussian stars.

    Stars are placed on a grid with jitter to ensure no aperture overlap.
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    image = np.zeros(shape, dtype=np.float64)
    y, x = np.mgrid[0:ny, 0:nx]

    # Place stars on a grid with jitter for well-separated sources
    margin = max(30, 7 * sigma)
    ncols = int(np.ceil(np.sqrt(nstars * nx / ny)))
    nrows = int(np.ceil(nstars / ncols))
    # Grid spacing
    dx = (nx - 2 * margin) / max(ncols, 1)
    dy = (ny - 2 * margin) / max(nrows, 1)

    xs_list = []
    ys_list = []
    for i in range(nstars):
        col = i % ncols
        row = i // ncols
        cx = margin + (col + 0.5) * dx + rng.uniform(-dx * 0.2, dx * 0.2)
        cy = margin + (row + 0.5) * dy + rng.uniform(-dy * 0.2, dy * 0.2)
        cx = np.clip(cx, margin, nx - margin)
        cy = np.clip(cy, margin, ny - margin)
        xs_list.append(cx)
        ys_list.append(cy)

    xs = np.array(xs_list)
    ys = np.array(ys_list)
    fluxes = np.full(nstars, flux)

    for x0, y0, f in zip(xs, ys, fluxes):
        image += gaussian2d(x, y, x0, y0, sigma, f)

    return image, xs, ys, fluxes


def measure_star_fluxes(image, wcs_in, xs_in, ys_in, wcs_out, aper_radius=15):
    """Measure aperture fluxes of stars in reprojected image.

    For zero-background images, just sum in aperture (no annulus needed).
    """
    sky = wcs_in.pixel_to_world(xs_in, ys_in)
    xo, yo = wcs_out.world_to_pixel(sky)

    ny, nx = image.shape
    fluxes = []

    yg, xg = np.mgrid[0:ny, 0:nx]
    for x0, y0 in zip(xo, yo):
        r2 = (xg - x0) ** 2 + (yg - y0) ** 2
        mask = r2 <= aper_radius ** 2
        if mask.any():
            fluxes.append(np.nansum(image[mask]))
        else:
            fluxes.append(np.nan)

    return np.array(fluxes), xo, yo


# ---------------------------------------------------------------------------
# Parallelization control
# ---------------------------------------------------------------------------
# Set to True to enable parallel reprojection (threaded).
# For reproject: uses block_size=(512,512) since block_size='auto' has a 64 MB
# threshold that prevents chunking for most practical image sizes.
# For stdpipe Lanczos: uses thread pool over row chunks.
USE_PARALLEL = True

# ---------------------------------------------------------------------------
# Reprojection wrappers with uniform interface
# All return (result, footprint, time_seconds)
# ---------------------------------------------------------------------------

def _parallel_kwargs():
    """Return reproject parallel kwargs based on USE_PARALLEL flag."""
    if USE_PARALLEL:
        return dict(parallel=True, block_size=(512, 512))
    return {}


def _run_reproject_interp(image, wcs_in, wcs_out, shape_out):
    hdr_out = make_header(wcs_out, shape_out)
    hdr_in = make_header(wcs_in, image.shape)
    t0 = time.time()
    result, footprint = reproject_interp(
        (image, hdr_in), hdr_out, shape_out=shape_out, **_parallel_kwargs()
    )
    dt = time.time() - t0
    return result, footprint, dt


def _run_reproject_adaptive(image, wcs_in, wcs_out, shape_out):
    hdr_out = make_header(wcs_out, shape_out)
    hdr_in = make_header(wcs_in, image.shape)
    t0 = time.time()
    result, footprint = reproject_adaptive(
        (image, hdr_in), hdr_out, shape_out=shape_out,
        conserve_flux=True, **_parallel_kwargs()
    )
    dt = time.time() - t0
    return result, footprint, dt


def _run_reproject_adaptive_nocons(image, wcs_in, wcs_out, shape_out):
    hdr_out = make_header(wcs_out, shape_out)
    hdr_in = make_header(wcs_in, image.shape)
    t0 = time.time()
    result, footprint = reproject_adaptive(
        (image, hdr_in), hdr_out, shape_out=shape_out,
        conserve_flux=False, **_parallel_kwargs()
    )
    dt = time.time() - t0
    return result, footprint, dt


def _run_reproject_exact(image, wcs_in, wcs_out, shape_out):
    hdr_out = make_header(wcs_out, shape_out)
    hdr_in = make_header(wcs_in, image.shape)
    t0 = time.time()
    result, footprint = reproject_exact(
        (image, hdr_in), hdr_out, shape_out=shape_out, **_parallel_kwargs()
    )
    dt = time.time() - t0
    return result, footprint, dt


def _run_reproject_bicubic(image, wcs_in, wcs_out, shape_out):
    hdr_out = make_header(wcs_out, shape_out)
    hdr_in = make_header(wcs_in, image.shape)
    t0 = time.time()
    result, footprint = reproject_interp(
        (image, hdr_in), hdr_out, shape_out=shape_out, order='bicubic',
        **_parallel_kwargs()
    )
    dt = time.time() - t0
    return result, footprint, dt


def _run_lanczos3(image, wcs_in, wcs_out, shape_out):
    hdr_out = make_header(wcs_out, shape_out)
    hdr_in = make_header(wcs_in, image.shape)
    t0 = time.time()
    result, footprint = reproject_interp_custom(
        (image, hdr_in), hdr_out, shape_out=shape_out,
        order=3, kernel='lanczos'
    )
    dt = time.time() - t0
    return result, footprint, dt


def _run_lanczos3_oversamp(image, wcs_in, wcs_out, shape_out):
    """Lanczos-3 with SWarp-style automatic oversampling + Jacobian."""
    hdr_out = make_header(wcs_out, shape_out)
    hdr_in = make_header(wcs_in, image.shape)
    t0 = time.time()
    result, footprint = reproject_interp_oversampled(
        (image, hdr_in), hdr_out, shape_out=shape_out,
        order=3, kernel='lanczos', oversamp=None  # auto
    )
    dt = time.time() - t0
    return result, footprint, dt


def _run_lanczos3_oversamp4(image, wcs_in, wcs_out, shape_out):
    """Lanczos-3 with fixed 4x oversampling + Jacobian."""
    hdr_out = make_header(wcs_out, shape_out)
    hdr_in = make_header(wcs_in, image.shape)
    t0 = time.time()
    result, footprint = reproject_interp_oversampled(
        (image, hdr_in), hdr_out, shape_out=shape_out,
        order=3, kernel='lanczos', oversamp=4
    )
    dt = time.time() - t0
    return result, footprint, dt


def _run_stdpipe_lanczos(image, wcs_in, wcs_out, shape_out):
    """stdpipe reproject_lanczos with flux conservation and optional parallelism."""
    hdr_in = make_header(wcs_in, image.shape)
    t0 = time.time()
    result = stdpipe_reproject_lanczos(
        [(image, hdr_in)],
        wcs=wcs_out,
        shape=shape_out,
        order=3,
        conserve_flux=True,
        parallel=USE_PARALLEL,
    )
    dt = time.time() - t0
    return result, None, dt


def _run_swarp(image, wcs_in, wcs_out, shape_out, verbose=False):
    """Run SWarp with noise added for valid weight computation.

    SWarp computes weights from background RMS. A zero-background image
    produces RMS~0, leading to zero weights and all-NaN output. We add
    small Gaussian noise (mean=0) that gives SWarp a valid RMS without
    biasing the flux.
    """
    rng = np.random.default_rng(12345)
    noise = rng.normal(0, 1.0, image.shape)
    hdr_in = make_header(wcs_in, image.shape)
    t0 = time.time()
    result = reproject_swarp(
        [(image + noise, hdr_in)],
        wcs=wcs_out,
        shape=shape_out,
        get_weights=False,
        verbose=verbose,
    )
    dt = time.time() - t0
    return result, None, dt


METHODS = OrderedDict([
    ('SWarp', _run_swarp),
    ('interp\n(bilinear)', _run_reproject_interp),
    ('interp\n(bicubic)', _run_reproject_bicubic),
    # ('Lanczos-3', _run_lanczos3),
    # ('Lanczos-3\n+oversamp', _run_lanczos3_oversamp),
    # ('Lanczos-3\n+oversamp4', _run_lanczos3_oversamp4),
    ('stdpipe\nLanczos-3', _run_stdpipe_lanczos),
    ('adaptive\n(conserve)', _run_reproject_adaptive),
    ('adaptive\n(no conserve)', _run_reproject_adaptive_nocons),
    ('exact', _run_reproject_exact),
])

SHORT_NAMES = {k: k.replace('\n', ' ') for k in METHODS}


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def run_test(name, image, wcs_in, wcs_out, shape_out, methods,
             xs=None, ys=None, fluxes=None, aper_radius=15,
             measure_stars=False, measure_total=True, scale_factor=1.0):
    """Generic test runner.

    When scale_factor != 1.0, star flux ratios are corrected for the
    known averaging-vs-summing difference: SB-conserving methods are
    expected to return star_flux_out/star_flux_in = 1/scale_factor^2,
    so we normalize by that factor to get the "corrected ratio" which
    should be ~1.0 for all correctly-behaving methods.
    """
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    total_in = np.nansum(image)
    area_ratio = scale_factor ** 2
    results = {}

    for mname, func in methods.items():
        out, fp, dt = func(image, wcs_in, wcs_out, shape_out)
        if out is None:
            print(f"  {mname:40s}: FAILED")
            continue

        entry = dict(time=dt)
        is_flux_cons = FLUX_CONSERVING.get(mname, True)

        if measure_total:
            total_out = np.nansum(out)
            # For flux-conserving: expect total_out = total_in
            # For SB-conserving:   expect total_out = total_in / area_ratio
            expected_total = total_in if is_flux_cons else total_in / area_ratio
            entry['total_ratio'] = total_out / expected_total

        if measure_stars and xs is not None:
            star_f, xo, yo = measure_star_fluxes(
                out, wcs_in, xs, ys, wcs_out, aper_radius=aper_radius
            )
            margin = aper_radius + 5
            ny, nx = shape_out
            inside = ((xo > margin) & (xo < nx - margin) &
                      (yo > margin) & (yo < ny - margin))
            if inside.sum() > 0:
                # Raw ratio
                raw_ratios = star_f[inside] / fluxes[inside]
                # Corrected: divide by expected ratio for this method type
                expected = 1.0 if is_flux_cons else 1.0 / area_ratio
                corrected_ratios = raw_ratios / expected
                entry['star_ratio'] = np.nanmean(corrected_ratios)
                entry['star_std'] = np.nanstd(corrected_ratios)
                entry['star_ratio_raw'] = np.nanmean(raw_ratios)
            else:
                entry['star_ratio'] = np.nan
                entry['star_std'] = np.nan
                entry['star_ratio_raw'] = np.nan

        results[mname] = entry

        # Print results
        parts = [f"  {mname:40s}:"]
        if 'total_ratio' in entry:
            parts.append(f"total(corr)={entry['total_ratio']:.6f}")
        if 'star_ratio' in entry:
            raw = entry.get('star_ratio_raw', np.nan)
            parts.append(f"star(raw)={raw:.6f}")
            parts.append(f"star(corr)={entry['star_ratio']:.6f}+/-{entry['star_std']:.6f}")
        parts.append(f"time={dt:.3f}s")
        print("  ".join(parts))

    return results


def test_identity(methods=METHODS):
    """Same WCS in and out — flux should be perfectly conserved."""
    shape = (256, 256)
    wcs = make_wcs(crpix=[128, 128], crval=[180.0, 45.0],
                   cdelt=[-1.0 / 3600, 1.0 / 3600], shape=shape)
    image, xs, ys, fluxes = make_star_field(shape, nstars=5, sigma=3.0, flux=10000.0)
    return run_test("Identity reprojection", image, wcs, wcs, shape, methods,
                    xs, ys, fluxes, aper_radius=20,
                    measure_stars=True, measure_total=True, scale_factor=1.0)


def test_shift(methods=METHODS):
    """Sub-pixel shift — flux should be conserved."""
    shape = (256, 256)
    wcs_in = make_wcs(crpix=[128, 128], crval=[180.0, 45.0],
                      cdelt=[-1.0 / 3600, 1.0 / 3600], shape=shape)
    wcs_out = make_wcs(crpix=[128.3, 128.7], crval=[180.0, 45.0],
                       cdelt=[-1.0 / 3600, 1.0 / 3600], shape=shape)
    image, xs, ys, fluxes = make_star_field(shape, nstars=5, sigma=3.0, flux=10000.0)
    return run_test("Sub-pixel shift (0.3, 0.7 px)", image, wcs_in, wcs_out, shape, methods,
                    xs, ys, fluxes, aper_radius=20,
                    measure_stars=True, measure_total=True, scale_factor=1.0)


def test_rotation(methods=METHODS):
    """15-degree rotation — per-star flux should be conserved (same pixel scale)."""
    shape = (256, 256)
    wcs_in = make_wcs(crpix=[128, 128], crval=[180.0, 45.0],
                      cdelt=[-1.0 / 3600, 1.0 / 3600], shape=shape)

    angle = np.deg2rad(15)
    cdelt = 1.0 / 3600
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = [128, 128]
    wcs_out.wcs.crval = [180.0, 45.0]
    wcs_out.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs_out.wcs.cd = np.array([
        [-cdelt * np.cos(angle), cdelt * np.sin(angle)],
        [-cdelt * np.sin(angle), -cdelt * np.cos(angle)],
    ])
    wcs_out.pixel_shape = shape[::-1]

    image, xs, ys, fluxes = make_star_field(shape, nstars=5, sigma=3.0, flux=10000.0)
    return run_test("15-degree rotation", image, wcs_in, wcs_out, shape, methods,
                    xs, ys, fluxes, aper_radius=20,
                    measure_stars=True, measure_total=True, scale_factor=1.0)


def test_rescale(methods=METHODS, scale_factor=2.0):
    """Pixel scale change — corrected star flux ratio should be ~1.0 for all methods."""
    shape_in = (256, 256)
    cdelt_in = 1.0 / 3600
    wcs_in = make_wcs(crpix=[128, 128], crval=[180.0, 45.0],
                      cdelt=[-cdelt_in, cdelt_in], shape=shape_in)

    cdelt_out = cdelt_in * scale_factor
    shape_out = (int(shape_in[0] / scale_factor), int(shape_in[1] / scale_factor))
    wcs_out = make_wcs(crpix=[shape_out[1] / 2, shape_out[0] / 2],
                       crval=[180.0, 45.0],
                       cdelt=[-cdelt_out, cdelt_out], shape=shape_out)

    image, xs, ys, fluxes = make_star_field(shape_in, nstars=5, sigma=3.0, flux=10000.0)
    return run_test(f"Rescale {scale_factor}x (per-star)", image, wcs_in, wcs_out,
                    shape_out, methods, xs, ys, fluxes,
                    aper_radius=max(8, int(20 / scale_factor) + 2),
                    measure_stars=True, measure_total=False,
                    scale_factor=scale_factor)


def test_uniform_field(methods=METHODS, scale_factor=1.5):
    """Uniform field under rescaling — reveals flux vs SB conservation behavior."""
    print(f"\n{'=' * 70}")
    print(f"  Uniform field rescaled by {scale_factor}x")
    print(f"  Flux-conserving: pixel value -> {scale_factor**2:.2f}")
    print(f"  SB-conserving:   pixel value -> 1.00")
    print(f"{'=' * 70}")

    shape_in = (256, 256)
    cdelt_in = 1.0 / 3600
    wcs_in = make_wcs(crpix=[128, 128], crval=[180.0, 45.0],
                      cdelt=[-cdelt_in, cdelt_in], shape=shape_in)

    cdelt_out = cdelt_in * scale_factor
    shape_out = (int(shape_in[0] / scale_factor), int(shape_in[1] / scale_factor))
    wcs_out = make_wcs(crpix=[shape_out[1] / 2, shape_out[0] / 2],
                       crval=[180.0, 45.0],
                       cdelt=[-cdelt_out, cdelt_out], shape=shape_out)

    image = np.ones(shape_in, dtype=np.float64)

    results = {}
    for mname, func in methods.items():
        out, fp, dt = func(image, wcs_in, wcs_out, shape_out)
        if out is None:
            print(f"  {mname:40s}: FAILED")
            continue

        margin = 5
        interior = out[margin:-margin, margin:-margin]
        valid = interior[np.isfinite(interior)]
        if len(valid) == 0:
            continue
        mean_val = np.mean(valid)

        is_flux_cons = FLUX_CONSERVING.get(mname, True)
        expected = scale_factor ** 2 if is_flux_cons else 1.0
        if abs(mean_val - expected) / expected < 0.02:
            behavior = 'flux-conserving' if is_flux_cons else 'SB-conserving'
        else:
            behavior = f'unexpected ({mean_val:.4f})'

        results[mname] = dict(pixel_value=mean_val, behavior=behavior, time=dt)
        print(f"  {mname:40s}: pixel_value={mean_val:.4f}  ({behavior})  time={dt:.3f}s")

    return results


def test_benchmark(methods=METHODS, size=1024):
    """Timing benchmark with a larger image."""
    shape = (size, size)
    wcs_in = make_wcs(crpix=[size // 2, size // 2], crval=[180.0, 45.0],
                      cdelt=[-1.0 / 3600, 1.0 / 3600], shape=shape)
    wcs_out = make_wcs(crpix=[size // 2 + 0.5, size // 2 + 0.5], crval=[180.0, 45.0],
                       cdelt=[-1.0 / 3600, 1.0 / 3600], shape=shape)

    image, xs, ys, fluxes = make_star_field(shape, nstars=10, sigma=3.0, flux=10000.0)
    return run_test(f"Benchmark {size}x{size}", image, wcs_in, wcs_out, shape, methods,
                    xs, ys, fluxes, aper_radius=20,
                    measure_stars=True, measure_total=True, scale_factor=1.0)


def test_fwhm_sweep(methods=METHODS):
    """Sweep FWHM from undersampled (0.5 px) to well-sampled (7 px).

    Uses sub-pixel shift reprojection (same pixel scale) to isolate the
    effect of source size on flux conservation accuracy.
    """
    fwhm_values = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]

    print(f"\n{'=' * 70}")
    print(f"  FWHM sweep: sub-pixel shift at varying FWHM")
    print(f"  FWHM values (pixels): {fwhm_values}")
    print(f"{'=' * 70}")

    shape = (256, 256)
    wcs_in = make_wcs(crpix=[128, 128], crval=[180.0, 45.0],
                      cdelt=[-1.0 / 3600, 1.0 / 3600], shape=shape)
    wcs_out = make_wcs(crpix=[128.3, 128.7], crval=[180.0, 45.0],
                       cdelt=[-1.0 / 3600, 1.0 / 3600], shape=shape)

    results = {}  # {method_name: {fwhm: corrected_ratio}}

    for fwhm in fwhm_values:
        sigma = fwhm / 2.355
        aper_radius = max(5, int(5 * sigma) + 2)
        image, xs, ys, fluxes = make_star_field(
            shape, nstars=5, sigma=sigma, flux=10000.0
        )

        print(f"\n  FWHM={fwhm:.1f} px (sigma={sigma:.2f}, aper={aper_radius} px):")
        for mname, func in methods.items():
            out, fp, dt = func(image, wcs_in, wcs_out, shape)
            if out is None:
                continue

            star_f, xo, yo = measure_star_fluxes(
                out, wcs_in, xs, ys, wcs_out, aper_radius=aper_radius
            )
            margin = aper_radius + 5
            ny, nx = shape
            inside = ((xo > margin) & (xo < nx - margin) &
                      (yo > margin) & (yo < ny - margin))
            if inside.sum() > 0:
                raw_ratios = star_f[inside] / fluxes[inside]
                ratio = np.nanmean(raw_ratios)
                std = np.nanstd(raw_ratios)
            else:
                ratio = np.nan
                std = np.nan

            results.setdefault(mname, {})[fwhm] = dict(
                ratio=ratio, std=std, time=dt
            )
            print(f"    {mname:40s}: ratio={ratio:.6f}+/-{std:.6f}")

    return results, fwhm_values


def test_fwhm_rescale_sweep(methods=METHODS, scale_factor=2.0):
    """Sweep FWHM under rescaling — the hardest test.

    Combines undersampled sources with pixel scale change.
    Corrects for SB vs flux conservation.
    """
    fwhm_values = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]

    print(f"\n{'=' * 70}")
    print(f"  FWHM sweep: {scale_factor}x rescale at varying FWHM")
    print(f"  FWHM values (pixels): {fwhm_values}")
    print(f"{'=' * 70}")

    shape_in = (256, 256)
    cdelt_in = 1.0 / 3600
    wcs_in = make_wcs(crpix=[128, 128], crval=[180.0, 45.0],
                      cdelt=[-cdelt_in, cdelt_in], shape=shape_in)

    cdelt_out = cdelt_in * scale_factor
    shape_out = (int(shape_in[0] / scale_factor), int(shape_in[1] / scale_factor))
    wcs_out = make_wcs(crpix=[shape_out[1] / 2, shape_out[0] / 2],
                       crval=[180.0, 45.0],
                       cdelt=[-cdelt_out, cdelt_out], shape=shape_out)

    area_ratio = scale_factor ** 2
    results = {}

    for fwhm in fwhm_values:
        sigma = fwhm / 2.355
        aper_radius = max(5, int(5 * sigma / scale_factor) + 2)
        image, xs, ys, fluxes = make_star_field(
            shape_in, nstars=5, sigma=sigma, flux=10000.0
        )

        print(f"\n  FWHM={fwhm:.1f} px (sigma={sigma:.2f}, aper={aper_radius} px):")
        for mname, func in methods.items():
            out, fp, dt = func(image, wcs_in, wcs_out, shape_out)
            if out is None:
                continue

            is_flux_cons = FLUX_CONSERVING.get(mname, True)
            star_f, xo, yo = measure_star_fluxes(
                out, wcs_in, xs, ys, wcs_out, aper_radius=aper_radius
            )
            margin = aper_radius + 3
            ny, nx = shape_out
            inside = ((xo > margin) & (xo < nx - margin) &
                      (yo > margin) & (yo < ny - margin))
            if inside.sum() > 0:
                raw_ratios = star_f[inside] / fluxes[inside]
                expected = 1.0 if is_flux_cons else 1.0 / area_ratio
                corrected = raw_ratios / expected
                ratio = np.nanmean(corrected)
                std = np.nanstd(corrected)
            else:
                ratio = np.nan
                std = np.nan

            results.setdefault(mname, {})[fwhm] = dict(
                ratio=ratio, std=std, time=dt
            )
            print(f"    {mname:40s}: corr_ratio={ratio:.6f}+/-{std:.6f}")

    return results, fwhm_values


def test_downscale_stars(methods=METHODS, scale_factor=3.0):
    """Per-star flux conservation under significant downscaling."""
    shape_in = (512, 512)
    cdelt_in = 1.0 / 3600
    wcs_in = make_wcs(crpix=[256, 256], crval=[180.0, 45.0],
                      cdelt=[-cdelt_in, cdelt_in], shape=shape_in)

    cdelt_out = cdelt_in * scale_factor
    shape_out = (int(shape_in[0] / scale_factor), int(shape_in[1] / scale_factor))
    wcs_out = make_wcs(crpix=[shape_out[1] / 2, shape_out[0] / 2],
                       crval=[180.0, 45.0],
                       cdelt=[-cdelt_out, cdelt_out], shape=shape_out)

    image, xs, ys, fluxes = make_star_field(shape_in, nstars=5, sigma=4.0, flux=50000.0)
    return run_test(f"Star flux {scale_factor}x downscale", image, wcs_in, wcs_out,
                    shape_out, methods, xs, ys, fluxes,
                    aper_radius=max(8, int(20 / scale_factor) + 2),
                    measure_stars=True, measure_total=False,
                    scale_factor=scale_factor)


# ---------------------------------------------------------------------------
# Summary and plotting
# ---------------------------------------------------------------------------

def make_summary_plot(all_results, fwhm_shift_results=None, fwhm_rescale_results=None,
                      fwhm_values=None):
    """Create a 6-panel summary figure."""
    method_names = list(METHODS.keys())
    n_methods = len(method_names)
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    method_colors = dict(zip(method_names, colors))
    _linestyles = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.', ':']
    _markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '*', 'X', 'P']
    linestyles = [_linestyles[i % len(_linestyles)] for i in range(n_methods)]
    markers = [_markers[i % len(_markers)] for i in range(n_methods)]

    fig = plt.figure(figsize=(18, 14))

    # Collect tests with star flux ratios
    star_tests = {}
    for test_name, results in all_results.items():
        if test_name == 'Uniform field':
            continue
        if not results:
            continue
        if any('star_ratio' in d for d in results.values()):
            star_tests[test_name] = results

    # ------------------------------------------------------------------
    # Top-left: Per-star corrected flux ratio (should be ~1.0 for all)
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(3, 2, 1)
    test_names = list(star_tests.keys())
    x = np.arange(len(test_names))
    bar_width = 0.08

    for i, method in enumerate(method_names):
        vals = []
        for test in test_names:
            data = star_tests[test].get(method, {})
            vals.append(data.get('star_ratio', np.nan))
        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax1.bar(x + offset, vals, bar_width,
                label=SHORT_NAMES.get(method, method).replace('\n', ' '),
                color=method_colors[method])

    ax1.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Corrected Flux Ratio')
    ax1.set_title('Per-Star Flux Conservation (FWHM=7 px)\n'
                   '(corrected for SB vs flux conservation)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=20, ha='right', fontsize=8)
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(axis='y', alpha=0.3)
    all_star_vals = []
    for test in test_names:
        for method in method_names:
            v = star_tests[test].get(method, {}).get('star_ratio', np.nan)
            if np.isfinite(v):
                all_star_vals.append(v)
    if all_star_vals:
        ymin = min(all_star_vals) - 0.005
        ymax = max(all_star_vals) + 0.005
        ax1.set_ylim(max(0.98, ymin), min(1.02, ymax))

    # ------------------------------------------------------------------
    # Top-right: Uniform field test
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(3, 2, 2)
    uniform_results = all_results.get('Uniform field', {})
    if uniform_results:
        methods_present = [m for m in method_names if m in uniform_results]
        vals = [uniform_results[m]['pixel_value'] for m in methods_present]
        behaviors = [uniform_results[m]['behavior'] for m in methods_present]
        bar_colors = [method_colors[m] for m in methods_present]
        labels = [SHORT_NAMES.get(m, m).replace('\n', ' ') for m in methods_present]

        bars = ax2.bar(range(len(methods_present)), vals, color=bar_colors)
        ax2.axhline(2.25, color='red', linestyle='--', alpha=0.7,
                     label='Flux-conserving (2.25)')
        ax2.axhline(1.0, color='blue', linestyle='--', alpha=0.7,
                     label='SB-conserving (1.0)')
        ax2.set_xticks(range(len(methods_present)))
        ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax2.set_ylabel('Output Pixel Value')
        ax2.set_title('Uniform Field Under 1.5x Rescale\n(input = 1.0 everywhere)')
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 3)
        for bar, val, beh in zip(bars, vals, behaviors):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ------------------------------------------------------------------
    # Middle-left: FWHM sweep — sub-pixel shift
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(3, 2, 3)
    if fwhm_shift_results and fwhm_values:
        for i, method in enumerate(method_names):
            mdata = fwhm_shift_results.get(method, {})
            if not mdata:
                continue
            fwhms = sorted(mdata.keys())
            ratios = [mdata[f]['ratio'] for f in fwhms]
            stds = [mdata[f]['std'] for f in fwhms]
            label = SHORT_NAMES.get(method, method).replace('\n', ' ')
            ax3.errorbar(fwhms, ratios, yerr=stds,
                         label=label, color=colors[i],
                         marker=markers[i], markersize=5,
                         linestyle=linestyles[i], linewidth=1.5,
                         capsize=3)

        ax3.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('FWHM (pixels)')
        ax3.set_ylabel('Star Flux Ratio')
        ax3.set_title('Flux Conservation vs FWHM\n(sub-pixel shift, same scale)')
        ax3.legend(fontsize=7, loc='best')
        ax3.grid(alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_xticks(fwhm_values)
        ax3.set_xticklabels([f'{f:.1f}' for f in fwhm_values])
        ax3.axvspan(0, 2.0, alpha=0.05, color='red', label='_undersampled')
        ax3.text(0.7, ax3.get_ylim()[0] + 0.001, 'undersampled',
                 fontsize=8, color='red', alpha=0.5, ha='center')

    # ------------------------------------------------------------------
    # Middle-right: FWHM sweep — 2x rescale (corrected)
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(3, 2, 4)
    if fwhm_rescale_results and fwhm_values:
        for i, method in enumerate(method_names):
            mdata = fwhm_rescale_results.get(method, {})
            if not mdata:
                continue
            fwhms = sorted(mdata.keys())
            ratios = [mdata[f]['ratio'] for f in fwhms]
            stds = [mdata[f]['std'] for f in fwhms]
            label = SHORT_NAMES.get(method, method).replace('\n', ' ')
            ax4.errorbar(fwhms, ratios, yerr=stds,
                         label=label, color=colors[i],
                         marker=markers[i], markersize=5,
                         linestyle=linestyles[i], linewidth=1.5,
                         capsize=3)

        ax4.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('FWHM (pixels)')
        ax4.set_ylabel('Corrected Star Flux Ratio')
        ax4.set_title('Flux Conservation vs FWHM\n(2x rescale, corrected for SB/flux)')
        ax4.legend(fontsize=7, loc='best')
        ax4.grid(alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_xticks(fwhm_values)
        ax4.set_xticklabels([f'{f:.1f}' for f in fwhm_values])
        ax4.axvspan(0, 2.0, alpha=0.05, color='red')
        ax4.text(0.7, ax4.get_ylim()[0] + 0.001, 'undersampled',
                 fontsize=8, color='red', alpha=0.5, ha='center')

    # ------------------------------------------------------------------
    # Bottom-left: Timing comparison
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(3, 2, 5)
    timing_data = {}
    for test_name, results in all_results.items():
        for method, data in results.items():
            if 'time' in data:
                timing_data.setdefault(method, []).append(data['time'])

    methods_with_times = [m for m in method_names if m in timing_data]
    mean_times = [np.mean(timing_data[m]) for m in methods_with_times]
    labels = [SHORT_NAMES.get(m, m).replace('\n', ' ') for m in methods_with_times]
    bar_colors = [method_colors[m] for m in methods_with_times]

    bars = ax5.barh(labels, mean_times, color=bar_colors)
    ax5.set_xlabel('Mean Time (seconds)')
    ax5.set_title('Average Reprojection Time')
    ax5.grid(axis='x', alpha=0.3)
    for bar, t in zip(bars, mean_times):
        ax5.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{t:.3f}s', va='center', fontsize=9)

    # ------------------------------------------------------------------
    # Bottom-right: Summary heatmap table
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')

    all_test_names = list(all_results.keys())
    col_labels = [SHORT_NAMES.get(m, m).replace('\n', ' ') for m in method_names]
    row_labels = all_test_names
    cell_text = []
    cell_colors = []

    for test_name in all_test_names:
        row_text = []
        row_colors = []
        results = all_results[test_name]
        for method in method_names:
            data = results.get(method, {})

            if test_name == 'Uniform field':
                pv = data.get('pixel_value', np.nan)
                beh = data.get('behavior', '?')
                if 'flux' in beh:
                    row_text.append(f'{pv:.2f}\n(flux-cons.)')
                    row_colors.append('#d4edda')  # green
                elif 'SB' in beh:
                    row_text.append(f'{pv:.2f}\n(SB-cons.)')
                    row_colors.append('#cce5ff')  # blue
                else:
                    row_text.append(f'{pv:.4f}')
                    row_colors.append('#fff3cd')  # yellow
            else:
                # Use corrected star_ratio, fall back to total_ratio
                ratio = data.get('star_ratio', data.get('total_ratio', np.nan))
                if np.isfinite(ratio):
                    err = abs(ratio - 1.0)
                    row_text.append(f'{ratio:.4f}')
                    if err < 0.002:
                        row_colors.append('#d4edda')  # green
                    elif err < 0.01:
                        row_colors.append('#fff3cd')  # yellow
                    else:
                        row_colors.append('#f8d7da')  # red
                else:
                    row_text.append('N/A')
                    row_colors.append('#e2e3e5')

        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax6.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.6)
    ax6.set_title('Summary: Corrected Flux Ratio\n'
                   'Green < 0.2%   Yellow < 1%   Red > 1%',
                   fontsize=10, pad=20)

    plt.tight_layout()
    plt.savefig('flux_conservation_benchmark.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to flux_conservation_benchmark.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Flux Conservation Benchmark: reproject_swarp vs reproject package")
    print("=" * 70)

    available = METHODS
    print(f"Methods: {[n.replace(chr(10), ' ') for n in available.keys()]}")

    all_results = OrderedDict()
    all_results['Identity'] = test_identity(available)
    all_results['Sub-pixel shift'] = test_shift(available)
    all_results['Rotation 15deg'] = test_rotation(available)
    all_results['Rescale 2x'] = test_rescale(available, scale_factor=2.0)
    all_results['Uniform field'] = test_uniform_field(available, scale_factor=1.5)
    all_results['Benchmark 1024'] = test_benchmark(available, size=1024)
    all_results['Star flux 3x down'] = test_downscale_stars(available, scale_factor=3.0)

    # FWHM sweep tests
    fwhm_shift_results, fwhm_values = test_fwhm_sweep(available)
    fwhm_rescale_results, _ = test_fwhm_rescale_sweep(available, scale_factor=2.0)

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE (corrected ratios — should all be ~1.0)")
    print(f"{'=' * 70}")
    hdr = (f"{'Test':<22s} {'Method':<35s} "
           f"{'Star(corr)':>12s} {'Total(corr)':>12s} {'Time':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for test_name, results in all_results.items():
        for method, data in results.items():
            sr = data.get('star_ratio', np.nan)
            tr = data.get('total_ratio', data.get('pixel_value', np.nan))
            sr_s = f"{sr:.6f}" if np.isfinite(sr) else "-"
            tr_s = f"{tr:.6f}" if np.isfinite(tr) else "-"
            t = f"{data.get('time', 0):.3f}s"
            mname = method.replace(chr(10), ' ')
            print(f"  {test_name:<20s} {mname:<35s} {sr_s:>12s} {tr_s:>12s} {t:>8s}")

    try:
        make_summary_plot(all_results, fwhm_shift_results, fwhm_rescale_results,
                          fwhm_values)
    except Exception as e:
        import traceback
        traceback.print_exc()
