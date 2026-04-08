"""
SFFT — Space-Frequency Fourier Transform image subtraction.

Implements spatially varying kernel fitting in a single global least-squares
solve, following the approach of Hu et al. (2022, ApJ, 936, 157).

The model is::

    science(x,y) = Σ_α Σ_β c_{α,β} · [P_β(x,y) · R_α(x,y)] + Σ_γ d_γ · P_γ(x,y)

where R_α is the reference shifted by kernel offset α, P_β are polynomial
basis functions encoding spatial variation, and c_{α,β} / d_γ are scalar
coefficients solved for globally.

The normal equations are assembled via batched BLAS matmuls over polynomial
term pairs, avoiding materializing the full design matrix.

A soft kernel-sum constraint enforces that Σ_α a_α(x,y) = f(x,y) where f is
a low-order polynomial modelling smooth flux-scale variation across the image.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SFFTResult:
    """Result of SFFT image subtraction."""

    diff: np.ndarray
    """Difference image (science - model)."""

    model: np.ndarray
    """Convolved reference + background model."""

    kernel_coeffs: np.ndarray
    """Kernel coefficients, shape (n_kernel, n_kpoly)."""

    bg_coeffs: np.ndarray
    """Background coefficients, shape (n_bgpoly,)."""

    kernel_shape: Tuple[int, int]
    """Kernel support size (ky, kx)."""

    kernel_poly_order: int
    """Polynomial order for kernel spatial variation."""

    bg_poly_order: int
    """Polynomial order for differential background."""

    flux_poly_order: int
    """Polynomial order for the kernel-sum (flux scale) constraint."""

    flux_poly_coeffs: np.ndarray
    """Fitted flux-scale polynomial coefficients."""

    n_iter: int
    """Number of sigma-clipping iterations performed."""

    rms: float
    """Final weighted RMS of residuals."""

    n_good: int
    """Number of good (unmasked, unclipped) pixels in final iteration."""


# ---------------------------------------------------------------------------
# Polynomial helpers
# ---------------------------------------------------------------------------


def _poly_terms_2d(x, y, order):
    """Triangular 2-D polynomial basis evaluated on coordinate arrays.

    Term ordering: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...

    Parameters
    ----------
    x, y : ndarray, same shape (any)
    order : int >= 0

    Returns
    -------
    terms : ndarray, shape (n_terms, *x.shape)
    """
    terms = []
    for total in range(order + 1):
        for px in range(total + 1):
            py = total - px
            terms.append(x**px * y**py)
    return np.array(terms, dtype=np.float64)


def _n_poly(order):
    """Number of terms in a 2-D triangular polynomial of given order."""
    return (order + 1) * (order + 2) // 2


def _norm_coords(ny, nx):
    """Pixel coordinate arrays normalized to [-1, 1]."""
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    cx, cy = 0.5 * (nx - 1), 0.5 * (ny - 1)
    sx = max(1.0, cx)
    sy = max(1.0, cy)
    return (xx - cx) / sx, (yy - cy) / sy


# ---------------------------------------------------------------------------
# Kernel offset helpers
# ---------------------------------------------------------------------------


def _kernel_offsets(ky, kx):
    """List of (dy, dx) offsets for a kernel of shape (ky, kx). Both must be odd."""
    if ky % 2 == 0 or kx % 2 == 0:
        raise ValueError("kernel dimensions must be odd, got (%d, %d)" % (ky, kx))
    hy, hx = ky // 2, kx // 2
    return [(dy, dx) for dy in range(-hy, hy + 1) for dx in range(-hx, hx + 1)]


def _shift_image(img, dy, dx):
    """Shift 2-D array by integer (dy, dx), zero-filling exposed edges."""
    ny, nx = img.shape
    out = np.zeros_like(img)
    sy0, sy1 = max(0, -dy), min(ny, ny - dy)
    dy0 = sy0 + dy
    sx0, sx1 = max(0, -dx), min(nx, nx - dx)
    dx0 = sx0 + dx
    if sy1 > sy0 and sx1 > sx0:
        out[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return out


# ---------------------------------------------------------------------------
# Core SFFT solve
# ---------------------------------------------------------------------------


def _assemble_normal_equations(
    reference, science, weight, kernel_shape, kernel_poly_order, bg_poly_order, x_norm, y_norm
):
    """Assemble normal equations H·θ = g via polynomial-pair batching.

    Instead of iterating over O(n_kernel²) offset pairs with small matmuls,
    iterates over O(n_kpoly²) polynomial term pairs with large BLAS matmuls.
    For typical parameters (7×7 kernel, poly=2), this is ~60× fewer iterations
    with much better BLAS utilization (49×49 matmuls vs 6×6).

    Parameters
    ----------
    reference : (ny, nx)
    science : (ny, nx)
    weight : (ny, nx)
    kernel_shape, kernel_poly_order, bg_poly_order : model specification
    x_norm, y_norm : (ny, nx) normalized coordinates

    Returns
    -------
    H : (n_total, n_total) normal matrix
    g : (n_total,) right-hand side
    """
    ky, kx = kernel_shape
    offsets = _kernel_offsets(ky, kx)
    n_kernel_pixels = ky * kx
    n_kpoly = _n_poly(kernel_poly_order)
    n_bgpoly = _n_poly(bg_poly_order)
    n_kernel_params = n_kernel_pixels * n_kpoly
    n_total = n_kernel_params + n_bgpoly

    ny, nx = reference.shape
    npix = ny * nx

    poly_k = _poly_terms_2d(x_norm, y_norm, kernel_poly_order)  # (n_kpoly, ny, nx)
    poly_bg = _poly_terms_2d(x_norm, y_norm, bg_poly_order)  # (n_bgpoly, ny, nx)

    # Flatten everything for matrix ops
    poly_k_flat = poly_k.reshape(n_kpoly, npix)  # (n_kpoly, npix)
    poly_bg_flat = poly_bg.reshape(n_bgpoly, npix)  # (n_bgpoly, npix)
    w_flat = weight.ravel()  # (npix,)
    s_flat = science.ravel()  # (npix,)

    H = np.zeros((n_total, n_total), dtype=np.float64)
    g = np.zeros(n_total, dtype=np.float64)

    # Pre-compute all shifted references: (n_kernel, npix)
    shifted_refs = np.empty((n_kernel_pixels, npix), dtype=np.float64)
    for a, (dy, dx) in enumerate(offsets):
        shifted_refs[a] = _shift_image(reference, dy, dx).ravel()

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        # === Background-background block ===
        w_poly_bg = poly_bg_flat * w_flat[np.newaxis, :]  # (n_bgpoly, npix)
        H[n_kernel_params:, n_kernel_params:] = w_poly_bg @ poly_bg_flat.T
        g[n_kernel_params:] = w_poly_bg @ s_flat

        # === Kernel RHS: g[α*n_kpoly + β] = Σ_pix w * P_β * R_α * S ===
        # For each poly term β: g_β[α] = Σ_pix (w * P_β * S) * R_α
        # = shifted_refs @ (w * P_β * S)  -- one matmul per β
        ws_flat = w_flat * s_flat
        for b in range(n_kpoly):
            wPS = ws_flat * poly_k_flat[b]  # (npix,)
            g_b = shifted_refs @ wPS  # (n_kernel,)
            # Scatter into g: g[α * n_kpoly + b] = g_b[α]
            g[b:n_kernel_params:n_kpoly] = g_b

        # === Kernel-background cross-block ===
        # H[α*n_kpoly+β, n_kernel_params+γ] = Σ_pix w * P_β * R_α * Q_γ
        # For each β: cross_β[α, γ] = Σ_pix (w * P_β * Q_γ) * R_α
        # = shifted_refs @ (w * P_β · poly_bg_flat.T)
        for b in range(n_kpoly):
            wPb = w_flat * poly_k_flat[b]  # (npix,)
            wPb_Q = poly_bg_flat * wPb[np.newaxis, :]  # (n_bgpoly, npix)
            cross_b = shifted_refs @ wPb_Q.T  # (n_kernel, n_bgpoly)
            # Scatter: H[α*n_kpoly+b, n_kernel_params:] = cross_b[α, :]
            for alpha in range(n_kernel_pixels):
                row = alpha * n_kpoly + b
                H[row, n_kernel_params : n_kernel_params + n_bgpoly] = cross_b[alpha]
                H[n_kernel_params : n_kernel_params + n_bgpoly, row] = cross_b[alpha]

        # === Kernel-kernel block (the main bottleneck) ===
        # H[α1*n_kpoly+β1, α2*n_kpoly+β2] = Σ_pix w * P_β1 * R_α1 * P_β2 * R_α2
        # For fixed (β1, β2):
        #   M[α1, α2] = Σ_pix (w * P_β1 * P_β2) * R_α1 * R_α2
        #             = (shifted_refs * √(w*P_β1*P_β2)) @ (shifted_refs * √(w*P_β1*P_β2)).T
        # Or directly: (shifted_refs * (w*P_β1)) @ (shifted_refs * P_β2).T
        # This is n_kpoly*(n_kpoly+1)/2 matmuls of size (n_kernel × npix) @ (npix × n_kernel)

        for b1 in range(n_kpoly):
            wPb1_refs = shifted_refs * (w_flat * poly_k_flat[b1])[np.newaxis, :]  # (n_kernel, npix)
            for b2 in range(b1, n_kpoly):
                Pb2_refs = shifted_refs * poly_k_flat[b2][np.newaxis, :]  # (n_kernel, npix)
                # M[α1, α2] = Σ_pix wPb1_refs[α1] * Pb2_refs[α2]
                M = wPb1_refs @ Pb2_refs.T  # (n_kernel, n_kernel) — single BLAS call

                # Scatter M into H:
                # H[α1*n_kpoly+b1, α2*n_kpoly+b2] = M[α1, α2]
                H[b1:n_kernel_params:n_kpoly, b2:n_kernel_params:n_kpoly] = M
                if b2 != b1:
                    H[b2:n_kernel_params:n_kpoly, b1:n_kernel_params:n_kpoly] = M.T

    return H, g


def _build_kernel_sum_constraint(kernel_shape, kernel_poly_order, flux_poly_order, x_norm, y_norm):
    """Build constraint matrix for kernel sum = flux polynomial.

    The kernel-sum at position (x,y) is:
        Σ_α a_α(x,y) = Σ_α Σ_β c_{α,β} · P_β(x,y)
                      = Σ_β [Σ_α c_{α,β}] · P_β(x,y)

    We want this to equal a flux polynomial:
        f(x,y) = Σ_γ f_γ · Q_γ(x,y)

    where Q_γ are polynomial terms up to flux_poly_order.

    If flux_poly_order <= kernel_poly_order, Q_γ ⊂ P_β, so the constraint is:
        Σ_α c_{α,β} = f_β   for β ≤ flux_poly_order terms
        Σ_α c_{α,β} = 0     for higher-order β terms

    This gives n_kpoly linear constraints on the kernel coefficients.

    Returns
    -------
    C : (n_constraints, n_total_params)
    d : (n_constraints,) — zero vector (flux coeffs become free variables
        handled via the constraint)
    n_flux_free : number of free flux polynomial coefficients
    """
    n_kernel_pixels = kernel_shape[0] * kernel_shape[1]
    n_kpoly = _n_poly(kernel_poly_order)
    n_flux = _n_poly(flux_poly_order)

    # We constrain: for each kernel poly term β with order > flux_poly_order,
    # the sum Σ_α c_{α,β} = 0.
    # For terms β within flux_poly_order, we don't constrain (those define
    # the free flux scale polynomial).

    # Count terms with total degree > flux_poly_order
    n_constrained = n_kpoly - n_flux
    if n_constrained <= 0:
        return None, None, n_flux

    # Build constraint rows
    # Parameter layout: [c_{0,0}, c_{0,1}, ..., c_{0,n_kpoly-1},
    #                    c_{1,0}, ..., c_{n_kernel-1, n_kpoly-1},
    #                    d_0, ..., d_{n_bgpoly-1}]
    # Total kernel params = n_kernel_pixels * n_kpoly
    # For poly term β: params at indices [α * n_kpoly + β for α in range(n_kernel_pixels)]

    n_total_kernel = n_kernel_pixels * n_kpoly
    C_rows = []

    # Identify which poly terms have total degree > flux_poly_order
    term_idx = 0
    for total in range(kernel_poly_order + 1):
        for px in range(total + 1):
            if total > flux_poly_order:
                # This term β should have Σ_α c_{α,β} = 0
                row = np.zeros(n_total_kernel, dtype=np.float64)
                for alpha in range(n_kernel_pixels):
                    row[alpha * n_kpoly + term_idx] = 1.0
                C_rows.append(row)
            term_idx += 1

    C = np.array(C_rows, dtype=np.float64)
    d = np.zeros(len(C_rows), dtype=np.float64)
    return C, d, n_flux


def _reconstruct_model(
    theta, reference, kernel_shape, kernel_poly_order, bg_poly_order, x_norm, y_norm
):
    """Reconstruct the model image from solved coefficients.

    model(x,y) = Σ_α [Σ_β c_{α,β} · P_β(x,y)] · R_α(x,y) + Σ_γ d_γ · Q_γ(x,y)

    Parameters
    ----------
    theta : (n_total,) solved coefficients
    reference : (ny, nx)
    kernel_shape, kernel_poly_order, bg_poly_order : model specification
    x_norm, y_norm : (ny, nx) normalized coordinates

    Returns
    -------
    model : (ny, nx)
    """
    ky, kx = kernel_shape
    offsets = _kernel_offsets(ky, kx)
    n_kpoly = _n_poly(kernel_poly_order)
    n_bgpoly = _n_poly(bg_poly_order)
    n_kernel_pixels = ky * kx

    poly_k = _poly_terms_2d(x_norm, y_norm, kernel_poly_order)  # (n_kpoly, ny, nx)
    poly_bg = _poly_terms_2d(x_norm, y_norm, bg_poly_order)  # (n_bgpoly, ny, nx)

    kernel_coeffs = theta[: n_kernel_pixels * n_kpoly].reshape(n_kernel_pixels, n_kpoly)
    bg_coeffs = theta[n_kernel_pixels * n_kpoly :]

    ny, nx = reference.shape
    npix = ny * nx

    # model = Σ_α a_α(x,y) · R_α(x,y) where a_α = Σ_β c_{α,β} · P_β
    # Compute a_α maps via matmul, then accumulate one offset at a time
    # to avoid materializing the full (n_kernel, npix) shifted_refs array.
    poly_k_flat = poly_k.reshape(n_kpoly, npix)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        a_maps = kernel_coeffs @ poly_k_flat  # (n_kernel, npix)

    model_flat = np.zeros(npix, dtype=np.float64)
    for alpha, (dy, dx) in enumerate(offsets):
        model_flat += a_maps[alpha] * _shift_image(reference, dy, dx).ravel()

    model = model_flat.reshape(ny, nx)

    # Background contribution (vectorized)
    model += np.tensordot(bg_coeffs, poly_bg, axes=(0, 0))

    return model


def _extract_flux_poly(theta, kernel_shape, kernel_poly_order, flux_poly_order):
    """Extract the fitted flux-scale polynomial from kernel coefficients.

    flux(x,y) = Σ_α a_α(x,y) = Σ_β [Σ_α c_{α,β}] · P_β(x,y)

    Only terms with total degree ≤ flux_poly_order contribute (higher
    terms are constrained to zero by the kernel-sum constraint).

    Returns
    -------
    flux_coeffs : (n_flux,) polynomial coefficients for the flux scale
    """
    n_kernel_pixels = kernel_shape[0] * kernel_shape[1]
    n_kpoly = _n_poly(kernel_poly_order)
    n_flux = _n_poly(flux_poly_order)

    kernel_coeffs = theta[: n_kernel_pixels * n_kpoly].reshape(n_kernel_pixels, n_kpoly)

    # Sum over all kernel pixels for each poly term
    sum_per_poly = kernel_coeffs.sum(axis=0)  # (n_kpoly,)

    # Return only the flux_poly_order terms
    return sum_per_poly[:n_flux]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve(
    image,
    template,
    mask=None,
    template_mask=None,
    err=None,
    kernel_shape=(7, 7),
    kernel_poly_order=2,
    bg_poly_order=2,
    flux_poly_order=1,
    flux_penalty=1e3,
    ridge=1e-6,
    sigma_clip=3.0,
    max_iter=5,
    verbose=False,
):
    """SFFT image subtraction with spatially varying kernel.

    Solves for a spatially varying convolution kernel and differential
    background in a single global least-squares problem. The kernel
    at each pixel is modelled as a delta-function basis with polynomial
    spatial variation. A soft constraint enforces that the kernel sum
    varies smoothly as a low-order polynomial (modelling flux-scale
    differences between science and template).

    Iterative sigma-clipping rejects outlier pixels (transients, cosmic
    rays, artifacts) that would otherwise bias the kernel solution.

    :param image: Science image as a 2-D NumPy array
    :param template: Template/reference image, same shape, aligned to science
    :param mask: Boolean mask where True = bad pixels to ignore, optional.
        If both science and template masks are needed, combine them before
        passing (``mask = science_mask | template_mask``).
    :param template_mask: Additional mask for template-only defects, optional.
        Merged with ``mask`` internally.
    :param err: Per-pixel error (standard deviation) map for weighting.
        If None, uniform weights are used. If set to True, a simple
        noise estimate is derived from the image.
    :param kernel_shape: (ky, kx) size of the convolution kernel, must be odd.
        Default (7, 7). Larger kernels handle bigger PSF differences but
        are slower.
    :param kernel_poly_order: Polynomial order for spatial variation of
        each kernel coefficient. Default 2 (quadratic). Higher orders
        capture more complex PSF variation but need more pixels.
    :param bg_poly_order: Polynomial order for the differential background
        model. Default 2.
    :param flux_poly_order: Polynomial order for the kernel-sum constraint
        (flux scale variation). Default 1 (linear gradient). Set to 0 for
        constant flux scale.
    :param flux_penalty: Penalty weight for the kernel-sum constraint.
        Default 1e3. Larger values enforce the constraint more strictly.
        Set to 0 to disable the constraint entirely.
    :param ridge: Tikhonov regularization parameter. Default 1e-6.
    :param sigma_clip: Sigma threshold for iterative outlier rejection.
        Default 3.0. Set to None or 0 to disable clipping.
    :param max_iter: Maximum number of sigma-clipping iterations. Default 5.
    :param verbose: If True, print progress. If callable, use as log function.
    :returns: :class:`SFFTResult` with difference image and all fit metadata
    """

    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # --- Input validation and preparation ---
    sci = np.asarray(image, dtype=np.float64)
    ref = np.asarray(template, dtype=np.float64)

    if sci.shape != ref.shape:
        raise ValueError(
            "science and template must have the same shape, got %s vs %s" % (sci.shape, ref.shape)
        )
    if sci.ndim != 2:
        raise ValueError("images must be 2-D arrays")

    ny, nx = sci.shape
    ky, kx = kernel_shape

    if ky % 2 == 0 or kx % 2 == 0:
        raise ValueError("kernel_shape must be odd, got (%d, %d)" % (ky, kx))

    if flux_poly_order > kernel_poly_order:
        raise ValueError(
            "flux_poly_order (%d) must be <= kernel_poly_order (%d)"
            % (flux_poly_order, kernel_poly_order)
        )

    # Replace NaN/Inf with 0 to prevent propagation through matmuls.
    # These pixels are masked via the weight map, so the values don't matter.
    nan_sci = ~np.isfinite(sci)
    nan_ref = ~np.isfinite(ref)
    if np.any(nan_sci) or np.any(nan_ref):
        sci = sci.copy()
        ref = ref.copy()
        sci[nan_sci] = 0.0
        ref[nan_ref] = 0.0

    n_kernel_pixels = ky * kx
    n_kpoly = _n_poly(kernel_poly_order)
    n_bgpoly = _n_poly(bg_poly_order)
    n_kernel_params = n_kernel_pixels * n_kpoly
    n_total = n_kernel_params + n_bgpoly

    n_fpoly = _n_poly(flux_poly_order)
    log(
        "SFFT: image %dx%d, kernel %dx%d, kernel_poly=%d (%d terms), "
        "bg_poly=%d (%d terms), flux_poly=%d (%d terms)"
        % (
            nx,
            ny,
            kx,
            ky,
            kernel_poly_order,
            n_kpoly,
            bg_poly_order,
            n_bgpoly,
            flux_poly_order,
            n_fpoly,
        )
    )
    log(
        "SFFT: %d kernel + %d background = %d total parameters"
        % (n_kernel_params, n_bgpoly, n_total)
    )

    # --- Build weight map ---
    # stdpipe convention: True = bad
    weight = np.ones((ny, nx), dtype=np.float64)

    if mask is not None:
        weight[np.asarray(mask, dtype=bool)] = 0.0

    if template_mask is not None:
        weight[np.asarray(template_mask, dtype=bool)] = 0.0

    # Mask original NaN/Inf pixels
    if np.any(nan_sci) or np.any(nan_ref):
        weight[nan_sci | nan_ref] = 0.0

    # Zero out edge pixels that are affected by kernel shifts
    hy, hx = ky // 2, kx // 2
    if hy > 0:
        weight[:hy, :] = 0.0
        weight[-hy:, :] = 0.0
    if hx > 0:
        weight[:, :hx] = 0.0
        weight[:, -hx:] = 0.0

    # Inverse-variance weighting if error map provided
    if err is True:
        # Simple noise estimate from image
        from astropy.stats import mad_std

        bg_rms = mad_std(sci[weight > 0]) if np.any(weight > 0) else 1.0
        err_map = np.full_like(sci, bg_rms)
        log("SFFT: estimated background RMS = %.2f" % bg_rms)
    elif err is not None:
        err_map = np.asarray(err, dtype=np.float64)
    else:
        err_map = None

    if err_map is not None:
        valid = (err_map > 0) & np.isfinite(err_map)
        weight[valid] *= 1.0 / err_map[valid] ** 2
        weight[~valid] = 0.0

    # --- Normalized coordinates ---
    x_norm, y_norm = _norm_coords(ny, nx)

    # --- Build kernel-sum constraint ---
    C_kernel, d_kernel, n_flux = _build_kernel_sum_constraint(
        kernel_shape, kernel_poly_order, flux_poly_order, x_norm, y_norm
    )

    if C_kernel is not None and flux_penalty > 0:
        log(
            "SFFT: kernel-sum constraint: %d equations, flux_poly_order=%d, "
            "penalty=%.1e" % (C_kernel.shape[0], flux_poly_order, flux_penalty)
        )
        # Pad constraint matrix to include background params (unconstrained)
        C_full = np.zeros((C_kernel.shape[0], n_total), dtype=np.float64)
        C_full[:, :n_kernel_params] = C_kernel
    else:
        C_full = None
        d_kernel = None

    # --- Iterative sigma-clipping solve ---
    do_clip = sigma_clip is not None and sigma_clip > 0 and max_iter > 1
    n_iter_done = 0
    final_rms = np.nan

    for iteration in range(max_iter):
        n_iter_done = iteration + 1
        n_good = int(np.sum(weight > 0))

        if n_good < n_total + 10:
            raise RuntimeError("Too few good pixels (%d) for %d parameters" % (n_good, n_total))

        # Assemble normal equations (memory-efficient: one offset at a time)
        log("SFFT: assembling normal equations (%d parameters)" % n_total)
        H, g = _assemble_normal_equations(
            ref, sci, weight, kernel_shape, kernel_poly_order, bg_poly_order, x_norm, y_norm
        )

        # Add kernel-sum penalty to normal equations
        if C_full is not None and flux_penalty > 0:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                H += flux_penalty * (C_full.T @ C_full)
                g += flux_penalty * (C_full.T @ d_kernel)

        # Solve with regularization
        H[np.diag_indices(n_total)] += ridge
        try:
            theta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            log("SFFT: WARNING - singular normal matrix, increasing ridge")
            H[np.diag_indices(n_total)] += 1e-3
            theta = np.linalg.solve(H, g)

        # Reconstruct model and compute residuals
        model = _reconstruct_model(
            theta, ref, kernel_shape, kernel_poly_order, bg_poly_order, x_norm, y_norm
        )
        residual = sci - model

        # Compute weighted RMS on good pixels
        good_mask = weight > 0
        if np.any(good_mask):
            final_rms = np.sqrt(np.mean(residual[good_mask] ** 2))
        else:
            final_rms = np.nan

        log(
            "SFFT: iteration %d/%d: n_good=%d, rms=%.4f"
            % (iteration + 1, max_iter, n_good, final_rms)
        )

        # Sigma-clipping
        if not do_clip or iteration == max_iter - 1:
            break

        if np.any(good_mask) and np.isfinite(final_rms) and final_rms > 0:
            # Use MAD for robust scale estimate
            abs_resid = np.abs(residual)
            robust_sigma = np.median(abs_resid[good_mask]) * 1.4826
            if robust_sigma > 0:
                clip_mask = abs_resid > sigma_clip * robust_sigma
                n_clipped = int(np.sum(clip_mask & good_mask))
                if n_clipped == 0:
                    log("SFFT: no pixels clipped, converged")
                    break
                weight[clip_mask] = 0.0
                log("SFFT: clipped %d pixels (%.3f%%)" % (n_clipped, 100.0 * n_clipped / n_good))
            else:
                break

    # --- Extract results ---
    kernel_coeffs = theta[:n_kernel_params].reshape(n_kernel_pixels, n_kpoly)
    bg_coeffs = theta[n_kernel_params:]
    flux_coeffs = _extract_flux_poly(theta, kernel_shape, kernel_poly_order, flux_poly_order)

    diff = sci - model
    n_good_final = int(np.sum(weight > 0))

    log(
        "SFFT: done. Final RMS=%.4f, n_good=%d (%.1f%%)"
        % (final_rms, n_good_final, 100.0 * n_good_final / (ny * nx))
    )
    log(
        "SFFT: flux scale polynomial coeffs: %s"
        % np.array2string(flux_coeffs, precision=4, separator=', ')
    )

    return SFFTResult(
        diff=diff,
        model=model,
        kernel_coeffs=kernel_coeffs,
        bg_coeffs=bg_coeffs,
        kernel_shape=kernel_shape,
        kernel_poly_order=kernel_poly_order,
        bg_poly_order=bg_poly_order,
        flux_poly_order=flux_poly_order,
        flux_poly_coeffs=flux_coeffs,
        n_iter=n_iter_done,
        rms=float(final_rms),
        n_good=n_good_final,
    )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def evaluate_kernel_at(result, x, y, image_shape):
    """Evaluate the spatially varying kernel at a single image position.

    :param result: :class:`SFFTResult` from :func:`solve`
    :param x: X pixel coordinate
    :param y: Y pixel coordinate
    :param image_shape: (ny, nx) of the original image
    :returns: 2-D array of shape ``result.kernel_shape``
    """
    ny, nx = image_shape
    x_n, y_n = _norm_coords(ny, nx)
    # Evaluate at single point
    xn = (x - 0.5 * (nx - 1)) / max(1.0, 0.5 * (nx - 1))
    yn = (y - 0.5 * (ny - 1)) / max(1.0, 0.5 * (ny - 1))

    poly_k = _poly_terms_2d(np.array(xn), np.array(yn), result.kernel_poly_order)  # (n_kpoly,)

    n_kernel_pixels = result.kernel_shape[0] * result.kernel_shape[1]
    kernel_vals = result.kernel_coeffs @ poly_k  # (n_kernel_pixels,)
    return kernel_vals.reshape(result.kernel_shape)


def evaluate_flux_scale(result, x, y, image_shape):
    """Evaluate the flux-scale polynomial at image position(s).

    :param result: :class:`SFFTResult` from :func:`solve`
    :param x: X coordinate(s), scalar or array
    :param y: Y coordinate(s), scalar or array
    :param image_shape: (ny, nx) of the original image
    :returns: Flux scale value(s), same shape as x/y
    """
    ny, nx = image_shape
    xn = (np.asarray(x, dtype=np.float64) - 0.5 * (nx - 1)) / max(1.0, 0.5 * (nx - 1))
    yn = (np.asarray(y, dtype=np.float64) - 0.5 * (ny - 1)) / max(1.0, 0.5 * (ny - 1))

    poly = _poly_terms_2d(xn, yn, result.flux_poly_order)
    return np.tensordot(result.flux_poly_coeffs, poly, axes=(0, 0))


def evaluate_background(result, x, y, image_shape):
    """Evaluate the differential background model at image position(s).

    :param result: :class:`SFFTResult` from :func:`solve`
    :param x: X coordinate(s), scalar or array
    :param y: Y coordinate(s), scalar or array
    :param image_shape: (ny, nx) of the original image
    :returns: Background value(s), same shape as x/y
    """
    ny, nx = image_shape
    xn = (np.asarray(x, dtype=np.float64) - 0.5 * (nx - 1)) / max(1.0, 0.5 * (nx - 1))
    yn = (np.asarray(y, dtype=np.float64) - 0.5 * (ny - 1)) / max(1.0, 0.5 * (ny - 1))

    poly = _poly_terms_2d(xn, yn, result.bg_poly_order)
    return np.tensordot(result.bg_coeffs, poly, axes=(0, 0))
