"""
Quad-hash astrometry solver (Python-only) with practical upgrades:

1) Multi-scale quad pools:
   - quads are partitioned into baseline-length *rank bins* (e.g. 6 bins)
   - consistent binning between detection and reference sets via shared edges
   - matching is done only within the same bin, reducing ambiguity

2) Two-stage hypothesis scoring with accuracy enhancements:
   - Stage 1: weighted scoring on a small subset with multi-probe hashing
   - Stage 2: mutual nearest-neighbor matching with weighted scoring
   - Iterative affine re-matching to grow the match set
   - Progressive sigma-clipping in final WCS refinement

Inputs
------
obj : astropy.table.Table with 'x','y' and either 'flux' or 'mag'
cat : astropy.table.Table with 'ra','dec','mag'
wcs_init : astropy.wcs.WCS (rough)

Outputs
-------
refined_wcs : astropy.wcs.WCS
match : astropy.table.Table
diagnostics : dict

Deps: numpy, scipy, astropy
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.wcs import WCS
from stdpipe.astrometry_wcs import fit_wcs_from_points


# -----------------------------
# Utilities
# -----------------------------

def _as_float_array(x) -> np.ndarray:
    a = np.asarray(x)
    return a.astype(np.float64, copy=False)

def _finite_mask(*cols) -> np.ndarray:
    m = np.ones(len(cols[0]), dtype=bool)
    for c in cols:
        c = np.asarray(c)
        m &= np.isfinite(c)
    return m

def _pick_brightest_obj(obj: Table, n: int) -> np.ndarray:
    if "flux" in obj.colnames:
        v = _as_float_array(obj["flux"])
        key = -v  # higher flux = brighter
    elif "mag" in obj.colnames:
        v = _as_float_array(obj["mag"])
        key = v   # lower mag = brighter
    else:
        raise ValueError("obj must contain 'flux' or 'mag'")
    idx = np.argsort(key)
    return idx[: min(n, len(idx))]

def _pick_brightest_cat(cat: Table, n: int) -> np.ndarray:
    v = _as_float_array(cat["mag"])
    idx = np.argsort(v)
    return idx[: min(n, len(idx))]

def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else np.nanstd(x)

def mag_signature(mags4: np.ndarray) -> Tuple[int, int, int, int]:
    mags4 = np.asarray(mags4, dtype=np.float64)
    if len(mags4) != 4:
        raise ValueError(f"Expected 4 magnitudes, got {len(mags4)}")
    return tuple(np.argsort(mags4).astype(int).tolist())

# -----------------------------
# TAN projection
# -----------------------------

def tan_project_deg(ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    ra = np.deg2rad(_as_float_array(ra_deg))
    dec = np.deg2rad(_as_float_array(dec_deg))
    ra0 = np.deg2rad(float(ra0_deg))
    dec0 = np.deg2rad(float(dec0_deg))

    dra = (ra - ra0 + np.pi) % (2 * np.pi) - np.pi

    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    sin_dec0, cos_dec0 = np.sin(dec0), np.cos(dec0)
    cos_dra = np.cos(dra)
    sin_dra = np.sin(dra)

    denom = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
    eps = 1e-12
    denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)

    u = (cos_dec * sin_dra) / denom
    v = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / denom
    return u, v

# -----------------------------
# Similarity transform (Umeyama)
# -----------------------------

def estimate_similarity_2d(A: np.ndarray, B: np.ndarray, allow_reflection: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.shape[1] != 2:
        raise ValueError("A and B must be (N,2) and same shape")

    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    X = A - mu_A
    Y = B - mu_B

    var_A = np.sum(X**2) / A.shape[0]
    if var_A <= 0:
        raise ValueError("Degenerate A variance")

    Sigma = (Y.T @ X) / A.shape[0]
    U, D, Vt = np.linalg.svd(Sigma)

    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    # If allow_reflection=True, accept the reflection as-is

    s = np.sum(D) / var_A
    t = mu_B - s * (R @ mu_A)
    return R, t, float(s)

def apply_similarity(xy: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return (s * (xy @ R.T)) + t

# -----------------------------
# Affine transform
# -----------------------------

def estimate_affine_2d(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit affine transform B = A @ M.T + t via least squares (6 DOF).

    More flexible than similarity (4 DOF) - handles shear and non-square pixels.
    Returns (M, t) where M is (2,2) linear part and t is (2,) translation.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = A.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for affine fit")
    A_aug = np.hstack([A, np.ones((n, 1))])  # (n, 3)
    cond = np.linalg.cond(A_aug)
    if cond > 1e12:
        raise ValueError(f"Degenerate affine fit (condition number {cond:.1e})")
    X, _, _, _ = np.linalg.lstsq(A_aug, B, rcond=None)
    M = X[:2].T  # (2, 2)
    t = X[2]     # (2,)
    if not (np.all(np.isfinite(M)) and np.all(np.isfinite(t))):
        raise ValueError("Degenerate affine fit (non-finite values)")
    return M, t

def apply_affine(xy: np.ndarray, M: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply affine transform: result = xy @ M.T + t

    Raises ValueError if the result contains non-finite values.
    """
    xy = np.asarray(xy, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = xy @ M.T + t
    if not np.all(np.isfinite(result)):
        raise ValueError("Affine transform produced non-finite values")
    return result

# -----------------------------
# Mutual nearest-neighbor matching
# -----------------------------

def _mutual_nearest_neighbor(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    radius: float,
    tree_b: Optional[cKDTree] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find mutual nearest-neighbor pairs between two point sets within radius.

    Returns (idx_a, idx_b) arrays of matched indices where each pair is
    the mutual closest match within the given radius.

    Args:
        xy_a: (N, 2) first point set
        xy_b: (M, 2) second point set
        radius: maximum matching distance
        tree_b: optional pre-built cKDTree for xy_b (avoids rebuilding)
    """
    if len(xy_a) == 0 or len(xy_b) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    tree_a = cKDTree(xy_a)
    if tree_b is None:
        tree_b = cKDTree(xy_b)

    # Forward: for each point in A, find nearest in B
    dist_fwd, nn_fwd = tree_b.query(xy_a, k=1, distance_upper_bound=radius)
    # Reverse: for each point in B, find nearest in A
    dist_rev, nn_rev = tree_a.query(xy_b, k=1, distance_upper_bound=radius)

    ok_fwd = np.isfinite(dist_fwd) & (dist_fwd < radius) & (nn_fwd < len(xy_b))
    a_ids = np.nonzero(ok_fwd)[0]
    b_ids = nn_fwd[a_ids].astype(int)

    # Mutual check: B's nearest neighbor back to A must be the same point
    mutual = nn_rev[b_ids] == a_ids
    return a_ids[mutual], b_ids[mutual]

# -----------------------------
# Quad generation & descriptors
# -----------------------------

def make_local_quads(points: np.ndarray, k: int = 8) -> List[Tuple[int, int, int, int]]:
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 4:
        return []
    k = int(min(k, max(3, n - 1)))

    tree = cKDTree(points)
    _, idxs = tree.query(points, k=k + 1)
    quads: List[Tuple[int, int, int, int]] = []
    for i in range(n):
        neigh = idxs[i, 1:]
        for a in range(len(neigh) - 2):
            for b in range(a + 1, len(neigh) - 1):
                for c in range(b + 1, len(neigh)):
                    quads.append((i, int(neigh[a]), int(neigh[b]), int(neigh[c])))
    return quads

def quad_descriptor(P: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Descriptor invariant to translation/rotation/scale.
    Returns (desc[4], baseline_len).
    """
    P = np.asarray(P, dtype=np.float64)
    d2 = np.sum((P[None, :, :] - P[:, None, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    if i == j:
        return np.full(4, np.nan), np.nan

    A = P[i]
    B = P[j]
    base = B - A
    L = float(np.hypot(base[0], base[1]))
    if L <= 0:
        return np.full(4, np.nan), np.nan

    e1 = base / L
    e2 = np.array([-e1[1], e1[0]], dtype=np.float64)

    idx = [k for k in range(4) if k not in (i, j)]
    C = P[idx[0]]
    D = P[idx[1]]

    c = np.array([np.dot(C - A, e1), np.dot(C - A, e2)], dtype=np.float64) / L
    d = np.array([np.dot(D - A, e1), np.dot(D - A, e2)], dtype=np.float64) / L

    pair = np.stack([c, d], axis=0)
    order = np.lexsort((pair[:, 1], pair[:, 0]))
    pair = pair[order]

    desc = np.array([pair[0, 0], pair[0, 1], pair[1, 0], pair[1, 1]], dtype=np.float64)
    return desc, L

# Phase 2 Optimization: Vectorized quad descriptor calculation
def quad_descriptor_batch(points: np.ndarray, quad_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized quad descriptor calculation for multiple quads.

    Args:
        points: (N, 2) array of point coordinates
        quad_indices: (M, 4) array of quad indices into points

    Returns:
        descs: (M, 4) array of descriptors
        lens: (M,) array of baseline lengths
    """
    M = quad_indices.shape[0]
    descs = np.full((M, 4), np.nan, dtype=np.float64)
    lens = np.full(M, np.nan, dtype=np.float64)

    if M == 0:
        return descs, lens

    # Gather all quad points: (M, 4, 2)
    P = points[quad_indices]

    # Compute pairwise distances for all quads: (M, 4, 4)
    diff = P[:, None, :, :] - P[:, :, None, :]  # (M, 4, 4, 2)
    d2 = np.sum(diff ** 2, axis=3)  # (M, 4, 4)

    # Find longest baseline for each quad
    flat_idx = np.argmax(d2.reshape(M, -1), axis=1)
    i_idx = flat_idx // 4
    j_idx = flat_idx % 4

    # Check for degenerate quads (i == j)
    valid = i_idx != j_idx

    # Process only valid quads
    if not np.any(valid):
        return descs, lens

    valid_mask = np.where(valid)[0]
    M_valid = len(valid_mask)

    # Extract baseline points for valid quads
    A = P[valid_mask, i_idx[valid_mask]]  # (M_valid, 2)
    B = P[valid_mask, j_idx[valid_mask]]  # (M_valid, 2)

    # Compute baseline vectors and lengths
    base = B - A  # (M_valid, 2)
    L = np.hypot(base[:, 0], base[:, 1])  # (M_valid,)

    # Filter out zero-length baselines
    valid_L = L > 0
    if not np.any(valid_L):
        return descs, lens

    final_mask = valid_mask[valid_L]
    M_final = len(final_mask)

    # Update for final valid quads
    A = A[valid_L]
    B = B[valid_L]
    base = base[valid_L]
    L = L[valid_L]
    i_idx_final = i_idx[final_mask]
    j_idx_final = j_idx[final_mask]
    P_final = P[final_mask]

    # Compute orthonormal basis
    e1 = base / L[:, None]  # (M_final, 2)
    e2 = np.stack([-e1[:, 1], e1[:, 0]], axis=1)  # (M_final, 2)

    # Find the other two points (C, D) for each quad (vectorized)
    # Create a mask for each quad indicating which indices are NOT i or j
    all_indices = np.arange(4)
    mask = np.ones((M_final, 4), dtype=bool)
    mask[np.arange(M_final), i_idx_final] = False
    mask[np.arange(M_final), j_idx_final] = False

    # Find the two other indices for each quad
    other_indices = np.where(mask)
    # Reshape to (M_final, 2) - two "other" indices per quad
    other_indices_reshaped = other_indices[1].reshape(M_final, 2)
    C_indices = other_indices_reshaped[:, 0]
    D_indices = other_indices_reshaped[:, 1]

    C = P_final[np.arange(M_final), C_indices]  # (M_final, 2)
    D = P_final[np.arange(M_final), D_indices]  # (M_final, 2)

    # Project C and D onto basis and normalize by L
    C_vec = C - A
    D_vec = D - A

    c_coords = np.stack([
        np.sum(C_vec * e1, axis=1),
        np.sum(C_vec * e2, axis=1)
    ], axis=1) / L[:, None]  # (M_final, 2)

    d_coords = np.stack([
        np.sum(D_vec * e1, axis=1),
        np.sum(D_vec * e2, axis=1)
    ], axis=1) / L[:, None]  # (M_final, 2)

    # Sort coordinates lexicographically for each quad (vectorized)
    # Stack c and d coordinates: (M_final, 2, 2) - 2 points × 2 coords
    pairs = np.stack([c_coords, d_coords], axis=1)  # (M_final, 2, 2)

    # Determine which point should come first (lexicographic order)
    # First compare x-coordinates ([:,0]), then y-coordinates ([:,1])
    swap = ((pairs[:, 0, 0] > pairs[:, 1, 0]) |
            ((pairs[:, 0, 0] == pairs[:, 1, 0]) & (pairs[:, 0, 1] > pairs[:, 1, 1])))

    # Swap where needed
    pairs[swap] = pairs[swap, ::-1, :]

    # Flatten to descriptor format: [x1, y1, x2, y2]
    descs[final_mask] = pairs.reshape(M_final, 4)
    lens[final_mask] = L

    return descs, lens

def quantize_desc(desc: np.ndarray, eps: float) -> Tuple[int, int, int, int]:
    q = np.floor(desc / eps + 0.5).astype(np.int64)
    return int(q[0]), int(q[1]), int(q[2]), int(q[3])

def reflect_desc(desc: np.ndarray) -> np.ndarray:
    """Reflect quad descriptor across its baseline (flip y) with lexicographic re-ordering."""
    x1, y1, x2, y2 = (float(desc[0]), float(desc[1]), float(desc[2]), float(desc[3]))
    y1 = -y1
    y2 = -y2
    if (x1 > x2) or (x1 == x2 and y1 > y2):
        x1, y1, x2, y2 = x2, y2, x1, y1
    return np.array([x1, y1, x2, y2], dtype=np.float64)

# Phase 1 Optimization: Pre-computed neighbor offsets cache
_NEIGHBOR_OFFSETS_CACHE: Dict[int, np.ndarray] = {}

def _get_neighbor_offsets(r: int) -> np.ndarray:
    """Get pre-computed neighbor offsets for given radius."""
    if r not in _NEIGHBOR_OFFSETS_CACHE:
        x = np.arange(-r, r + 1, dtype=np.int32)
        # Create meshgrid and reshape to (n_neighbors, 4)
        grid = np.meshgrid(x, x, x, x, indexing='ij')
        offsets = np.stack([g.ravel() for g in grid], axis=1)
        _NEIGHBOR_OFFSETS_CACHE[r] = offsets
    return _NEIGHBOR_OFFSETS_CACHE[r]

def neighbors_bins(key: Tuple[int, int, int, int], r: int = 1) -> Iterable[Tuple[int, int, int, int]]:
    """Generate neighbor bins using optimized pre-computed offsets."""
    offsets = _get_neighbor_offsets(r)
    key_arr = np.array(key, dtype=np.int32)
    neighbors = key_arr + offsets
    # Return as tuples of Python ints for compatibility (important for dict keys)
    for neighbor in neighbors:
        yield tuple(int(x) for x in neighbor)

def multiprobe_desc_keys(desc: np.ndarray, eps: float, threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """Multi-probe hashing: return the primary quantized key plus keys for nearby
    bins where the descriptor is close to a bin boundary.

    Instead of searching all (2r+1)^4 = 81 neighboring bins, this probes only
    the bins that the descriptor might fall into due to quantization noise.
    Typically returns 1-8 keys instead of 81.

    Args:
        desc: 4D descriptor array
        eps: quantization step size
        threshold: fraction of eps within which a boundary is considered "near"
    """
    q = desc / eps + 0.5
    base = np.floor(q).astype(np.int64)
    frac = q - base  # fractional part in [0, 1)

    primary = tuple(int(x) for x in base)
    keys = [primary]

    # Find dimensions near bin boundaries
    near_dims = []
    offsets_per_dim = {}

    for d in range(4):
        if frac[d] < threshold:
            near_dims.append(d)
            offsets_per_dim[d] = -1
        elif frac[d] > (1.0 - threshold):
            near_dims.append(d)
            offsets_per_dim[d] = 1

    # Single-dimension probes
    for d in near_dims:
        alt = list(base)
        alt[d] += offsets_per_dim[d]
        keys.append(tuple(int(x) for x in alt))

    # Two-dimension combination probes
    for i in range(len(near_dims)):
        for j in range(i + 1, len(near_dims)):
            d1, d2 = near_dims[i], near_dims[j]
            alt = list(base)
            alt[d1] += offsets_per_dim[d1]
            alt[d2] += offsets_per_dim[d2]
            keys.append(tuple(int(x) for x in alt))

    return keys

def compute_bin_edges(lengths: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    """Compute quantile-based bin edges from a set of baseline lengths.

    Returns array of (n_bins+1) edge values, or None if binning not possible.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    if len(lengths) == 0 or n_bins <= 1:
        return None
    qs = np.quantile(lengths, q=np.linspace(0, 1, n_bins + 1))
    for i in range(1, len(qs)):
        if qs[i] <= qs[i - 1]:
            qs[i] = np.nextafter(qs[i - 1], np.inf)
    return qs

def baseline_rank_bins(lengths: np.ndarray, n_bins: int, edges: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Assign each length to a *rank* bin based on quantiles.
    If edges are provided, use those instead of computing from the data.
    This allows consistent binning between detection and reference sets.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    n = len(lengths)
    if n == 0:
        return np.zeros(0, dtype=np.int16)
    if n_bins <= 1:
        return np.zeros(n, dtype=np.int16)
    if edges is not None:
        b = np.searchsorted(edges[1:-1], lengths, side="right")
        return np.clip(b, 0, n_bins - 1).astype(np.int16)
    qs = np.quantile(lengths, q=np.linspace(0, 1, n_bins + 1))
    # make strictly increasing to avoid pathological equal-quantile edges
    for i in range(1, len(qs)):
        if qs[i] <= qs[i - 1]:
            qs[i] = np.nextafter(qs[i - 1], np.inf)
    b = np.searchsorted(qs[1:-1], lengths, side="right")
    return b.astype(np.int16)

@dataclass
class QuadEntry:
    idxs: Tuple[int, int, int, int]
    desc: np.ndarray
    bin_id: int
    mags: Optional[Tuple[float, float, float, float]] = None

def build_quad_hash_multiscale(
    points: np.ndarray,
    mags: Optional[np.ndarray],
    k: int,
    eps: float,
    n_scale_bins: int,
    allow_reflection: bool = True,
    bin_edges: Optional[np.ndarray] = None,
) -> Dict[Tuple[int, Tuple[int, int, int, int]], List[QuadEntry]]:
    """
    Hash key: (scale_bin_id, quantized_descriptor)

    If bin_edges is provided, use those for baseline binning instead of
    computing quantiles from the data. This enables consistent binning
    between detection and reference sets.

    Phase 2 Optimization: Uses vectorized quad descriptor calculation.
    """
    quads = make_local_quads(points, k=k)
    if not quads:
        return {}

    # Phase 2: Vectorized descriptor calculation
    quad_array = np.array(quads, dtype=np.int32)
    descs, lens = quad_descriptor_batch(points, quad_array)

    # Phase 1: Use boolean indexing instead of list comprehension
    ok = np.isfinite(descs).all(axis=1) & np.isfinite(lens) & (lens > 0)
    quads_ok = quad_array[ok]  # Keep as numpy array
    descs_ok = descs[ok]
    lens_ok = lens[ok]

    if len(quads_ok) == 0:
        return {}

    bins = baseline_rank_bins(lens_ok, n_bins=n_scale_bins, edges=bin_edges)

    # Phase 1: Pre-allocate and use numpy operations
    H: Dict[Tuple[int, Tuple[int, int, int, int]], List[QuadEntry]] = {}
    for i in range(len(quads_ok)):
        q = tuple(quads_ok[i])  # Convert to tuple for QuadEntry
        desc = descs_ok[i]
        b = int(bins[i])

        me = None
        if mags is not None:
            me = tuple(float(mags[idx]) for idx in q)
        entry = QuadEntry(idxs=q, desc=desc, bin_id=b, mags=me)

        # Insert descriptor and its reflected counterpart to handle parity flips
        key = (b, quantize_desc(desc, eps))
        H.setdefault(key, []).append(entry)

        if allow_reflection:
            desc_ref = reflect_desc(desc)
            key_ref = (b, quantize_desc(desc_ref, eps))
            if key_ref != key:
                H.setdefault(key_ref, []).append(entry)

    return H

class _PatternMatchFailed(RuntimeError):
    """Internal sentinel for pattern matching failure (used by adaptive retry)."""
    pass


def _auto_match_resolution(det_xy: np.ndarray, ref_uv: np.ndarray) -> Optional[float]:
    """Compute matching resolution from source density (SCAMP-inspired).

    Uses the mean inter-source spacing as confusion limit:
        matchresol = sqrt(field_area / n_sources)

    Returns the confusion radius in pixels, or None if it can't be computed.
    """
    # Use the smaller of the two sets (like SCAMP's cross-section approach)
    n = min(len(det_xy), len(ref_uv))
    if n < 4:
        return None

    # Estimate field area from convex hull or bounding box of detections
    x_range = det_xy[:, 0].max() - det_xy[:, 0].min()
    y_range = det_xy[:, 1].max() - det_xy[:, 1].min()
    field_area = x_range * y_range

    if field_area <= 0:
        return None

    # Mean area per source → confusion radius
    mean_area = field_area / n
    confusion_radius = np.sqrt(mean_area)

    return confusion_radius


# -----------------------------
# Solver
# -----------------------------

@dataclass
class AstrometryConfig:
    # Selection sizes
    n_det: int = 180
    n_ref: int = 380

    # Quads
    neighbor_k: int = 8
    eps_desc: float = 0.015
    bin_neighbor_radius: int = 1
    n_scale_bins: int = 6

    allow_reflection: bool = True
    use_mag_signature: bool = True

    # WCS prior constraints (use initial WCS to filter hypotheses)
    use_wcs_prior: bool = True
    scale_tolerance: float = 0.30  # fractional, e.g. 0.30 = +/-30%
    enforce_parity: bool = True

    # Two-stage scoring
    stage1_n_det: int = 50              # cheap scoring uses only brightest subset of selected detections
    stage1_radius_arcsec: float = 20.0  # loose
    stage2_radius_arcsec: float = 8.0   # tighter
    top_k_hypotheses: int = 60          # keep best hypotheses from stage 1
    max_quads_tested: int = 5000        # max detection quads to try
    max_candidates_per_bucket: int = 60 # cap ref candidates pulled per hash bucket

    # Multi-probe hashing: probe only nearby bins instead of all (2r+1)^4
    use_multiprobe: bool = True

    # Final fit
    sip_degree: int = 3
    refine_clip_sigma: float = 4.0
    refine_clip_sigma_start: float = 5.0    # progressive clipping: start value
    refine_min_match_fraction: float = 0.5  # keep at least this fraction of initial matches
    refine_max_iter: int = 5
    refine_rematch_iters: int = 2           # iterative affine re-matching rounds
    # Optional expanded refinement pool (use many more objects for final fit)
    refine_use_all: bool = False
    refine_n_det: Optional[int] = None
    refine_n_ref: Optional[int] = None
    refine_match_radius_arcsec: Optional[float] = None

    # Auto matching resolution: compute stage radii from source density
    auto_match_resolution: bool = True

    # Adaptive source count: retry with more sources if matching fails
    adaptive_n_retry: int = 2          # number of retry doublings (0 = disabled)
    adaptive_min_inliers: int = 12     # minimum inliers to accept without retry


class QuadHashAstrometry:
    def __init__(self, config: AstrometryConfig = AstrometryConfig()):
        self.cfg = config

    def _pattern_match(
        self,
        det_xy: np.ndarray,
        det_mag: np.ndarray,
        ref_uv: np.ndarray,
        ref_mag: np.ndarray,
        wcs_init: WCS,
        pixel_scale_arcsec: float,
        cfg: Optional['AstrometryConfig'] = None,
    ) -> Tuple[list, list, dict]:
        """Run quad-hash pattern matching (stages 1 & 2) + affine re-matching.

        Returns (pairs, top_hyp, best) where pairs is list of (det_idx, ref_idx).
        Raises _PatternMatchFailed if matching fails.
        """
        if cfg is None:
            cfg = self.cfg
        # WCS prior: in pixel space expected scale ≈ 1.0, compute parity from WCS
        expected_scale = None
        expected_parity = None  # sign of determinant
        if cfg.use_wcs_prior:
            expected_scale = 1.0  # Both sets in pixel space
            try:
                crpix = np.asarray(wcs_init.wcs.crpix, dtype=np.float64)
                if crpix.size == 2 and np.all(np.isfinite(crpix)):
                    px = np.array([crpix[0], crpix[0] + 1.0, crpix[0]], dtype=np.float64)
                    py = np.array([crpix[1], crpix[1], crpix[1] + 1.0], dtype=np.float64)
                    ra_s, dec_s = wcs_init.all_pix2world(px, py, 1)
                    x_s, y_s = wcs_init.all_world2pix(ra_s, dec_s, 0)
                    J = np.array([
                        [x_s[1] - x_s[0], x_s[2] - x_s[0]],
                        [y_s[1] - y_s[0], y_s[2] - y_s[0]],
                    ], dtype=np.float64)
                    det_j = float(np.linalg.det(J))
                    if np.isfinite(det_j) and det_j != 0:
                        expected_parity = 1.0 if det_j > 0 else -1.0
            except Exception:
                expected_parity = None

        # --- Consistent baseline binning ---
        ref_quads_raw = make_local_quads(ref_uv, k=cfg.neighbor_k)
        ref_bin_edges = None
        if ref_quads_raw:
            ref_qa = np.array(ref_quads_raw, dtype=np.int32)
            _, ref_lens_raw = quad_descriptor_batch(ref_uv, ref_qa)
            ok_ref = np.isfinite(ref_lens_raw) & (ref_lens_raw > 0)
            if np.any(ok_ref):
                ref_bin_edges = compute_bin_edges(ref_lens_raw[ok_ref], cfg.n_scale_bins)

        # Reference multiscale hash
        ref_hash = build_quad_hash_multiscale(
            ref_uv, ref_mag,
            k=cfg.neighbor_k,
            eps=cfg.eps_desc,
            n_scale_bins=cfg.n_scale_bins,
            allow_reflection=cfg.allow_reflection,
            bin_edges=ref_bin_edges,
        )

        # Detection quads
        det_quads = make_local_quads(det_xy, k=cfg.neighbor_k)
        if not det_quads:
            raise _PatternMatchFailed("Not enough detections for quad matching.")

        det_quad_array = np.array(det_quads, dtype=np.int32)
        det_descs, det_lens = quad_descriptor_batch(det_xy, det_quad_array)

        okq = np.isfinite(det_descs).all(axis=1) & np.isfinite(det_lens) & (det_lens > 0)
        det_quads = det_quad_array[okq]
        det_descs = det_descs[okq]
        det_lens = det_lens[okq]

        if ref_bin_edges is not None:
            det_bins = baseline_rank_bins(det_lens, n_bins=cfg.n_scale_bins, edges=ref_bin_edges)
        else:
            det_bins = baseline_rank_bins(det_lens, n_bins=cfg.n_scale_bins)

        det_quads_quantized = np.array([
            quantize_desc(desc, cfg.eps_desc) for desc in det_descs
        ], dtype=object)

        ref_tree = cKDTree(ref_uv)

        # Stage radii in pixels
        r1 = cfg.stage1_radius_arcsec / pixel_scale_arcsec
        r2 = cfg.stage2_radius_arcsec / pixel_scale_arcsec

        # Stage1 detection subset
        order_det = np.argsort(det_mag)
        det_stage1_ids = order_det[: min(cfg.stage1_n_det, len(order_det))]
        det_xy_stage1 = det_xy[det_stage1_ids]

        top_hyp: List[Tuple[float, np.ndarray, np.ndarray, float]] = []

        rng = np.random.default_rng(12345)

        q_order = np.arange(len(det_quads))
        rng.shuffle(q_order)
        q_order = q_order[: min(cfg.max_quads_tested, len(q_order))]

        def _try_insert(score: float, R: np.ndarray, t: np.ndarray, s: float):
            nonlocal top_hyp
            if len(top_hyp) < cfg.top_k_hypotheses:
                top_hyp.append((score, R, t, s))
                return
            worst_i = int(np.argmin([h[0] for h in top_hyp]))
            if score > top_hyp[worst_i][0]:
                top_hyp[worst_i] = (score, R, t, s)

        # --- Stage 1 ---
        for qi in q_order:
            q = det_quads[qi]
            bin_id = int(det_bins[qi])

            kdesc = det_quads_quantized[qi]

            candidates: List[QuadEntry] = []
            if cfg.use_multiprobe:
                for kdesc2 in multiprobe_desc_keys(det_descs[qi], cfg.eps_desc):
                    kk = (bin_id, kdesc2)
                    if kk in ref_hash:
                        candidates.extend(ref_hash[kk])
            else:
                for kdesc2 in neighbors_bins(kdesc, r=cfg.bin_neighbor_radius):
                    kk = (bin_id, kdesc2)
                    if kk in ref_hash:
                        candidates.extend(ref_hash[kk])

            if not candidates:
                continue

            det_sig = mag_signature(det_mag[q]) if cfg.use_mag_signature else None

            rng.shuffle(candidates)
            for ce in candidates[: cfg.max_candidates_per_bucket]:
                if det_sig is not None and ce.mags is not None:
                    if mag_signature(np.array(ce.mags)) != det_sig:
                        continue

                P = det_xy[q]
                Q = ref_uv[list(ce.idxs)]

                try:
                    R, t, s = estimate_similarity_2d(P, Q, allow_reflection=cfg.allow_reflection)
                except Exception:
                    continue

                if expected_scale is not None and cfg.scale_tolerance is not None:
                    tol = float(cfg.scale_tolerance)
                    if tol >= 0:
                        if not (expected_scale * (1 - tol) <= s <= expected_scale * (1 + tol)):
                            continue
                if (expected_parity is not None and cfg.enforce_parity
                        and cfg.allow_reflection):
                    detR = float(np.linalg.det(R))
                    if not np.isfinite(detR) or np.sign(detR) != expected_parity:
                        continue

                det_uv1 = apply_similarity(det_xy_stage1, R, t, s)
                dist, nn = ref_tree.query(det_uv1, k=1, distance_upper_bound=r1)
                ok = np.isfinite(dist) & (dist < r1) & (nn < len(ref_uv))
                if not np.any(ok):
                    continue

                weights = 1.0 - (dist[ok] / r1) ** 2
                score = float(np.sum(weights))

                _try_insert(score, R, t, s)

        if not top_hyp:
            raise _PatternMatchFailed(
                "Pattern matching failed at stage 1. "
                "Try increasing n_det/n_ref, eps_desc, or stage1_radius_arcsec.")

        # --- Stage 2 ---
        top_hyp.sort(key=lambda x: x[0], reverse=True)

        best = {"score": -1.0, "inliers": -1, "R": None, "t": None, "s": None, "pairs": None}

        for (score1, R, t, s) in top_hyp:
            if expected_scale is not None and cfg.scale_tolerance is not None:
                tol = float(cfg.scale_tolerance)
                if tol >= 0:
                    if not (expected_scale * (1 - tol) <= s <= expected_scale * (1 + tol)):
                        continue
            if (expected_parity is not None and cfg.enforce_parity
                    and cfg.allow_reflection):
                detR = float(np.linalg.det(R))
                if not np.isfinite(detR) or np.sign(detR) != expected_parity:
                    continue

            det_uv = apply_similarity(det_xy, R, t, s)

            det_ids, ref_ids = _mutual_nearest_neighbor(det_uv, ref_uv, r2, tree_b=ref_tree)

            if len(det_ids) == 0:
                continue

            dists = np.hypot(
                det_uv[det_ids, 0] - ref_uv[ref_ids, 0],
                det_uv[det_ids, 1] - ref_uv[ref_ids, 1]
            )
            weights = 1.0 - (dists / r2) ** 2
            score = float(np.sum(weights))

            pairs = list(zip(det_ids.tolist(), ref_ids.tolist()))

            if score > best["score"]:
                best = {"score": score, "inliers": len(pairs), "R": R, "t": t, "s": s, "pairs": pairs}

        if best["inliers"] < 8:
            raise _PatternMatchFailed(
                f"Pattern matching failed at stage 2 (best inliers={best['inliers']}). "
                f"Try loosening stage2_radius_arcsec or increasing top_k_hypotheses.")

        # --- Iterative affine re-matching ---
        pairs = best["pairs"]
        r_rematch = r2

        for rematch_it in range(cfg.refine_rematch_iters):
            if len(pairs) < 6:
                break

            pairs_arr = np.array(pairs, dtype=np.int32)
            A_pts = det_xy[pairs_arr[:, 0]]
            B_pts = ref_uv[pairs_arr[:, 1]]

            try:
                M_affine, t_affine = estimate_affine_2d(A_pts, B_pts)
                det_uv_affine = apply_affine(det_xy, M_affine, t_affine)
            except Exception:
                break

            r_iter = r_rematch * (0.8 if rematch_it > 0 else 1.0)

            new_det_ids, new_ref_ids = _mutual_nearest_neighbor(
                det_uv_affine, ref_uv, r_iter, tree_b=ref_tree
            )

            if len(new_det_ids) < max(8, int(len(pairs) * 0.6)):
                break

            pairs = list(zip(new_det_ids.tolist(), new_ref_ids.tolist()))

        return pairs, top_hyp, best

    def refine(self, obj: Table, cat: Table, wcs_init: WCS) -> Tuple[WCS, Table, dict]:
        if not all(k in obj.colnames for k in ("x", "y")):
            raise ValueError("obj must contain 'x' and 'y'")
        if not all(k in cat.colnames for k in ("ra", "dec", "mag")):
            raise ValueError("cat must contain 'ra','dec','mag'")

        x = _as_float_array(obj["x"])
        y = _as_float_array(obj["y"])
        if "mag" in obj.colnames:
            m_obj = _as_float_array(obj["mag"])
        elif "flux" in obj.colnames:
            f = _as_float_array(obj["flux"])
            f = np.where(f > 0, f, np.nan)
            m_obj = -2.5 * np.log10(f)
        else:
            raise ValueError("obj must contain 'flux' or 'mag'")

        ra = _as_float_array(cat["ra"])
        dec = _as_float_array(cat["dec"])
        m_cat = _as_float_array(cat["mag"])

        m0 = _finite_mask(x, y, m_obj)
        x, y, m_obj = x[m0], y[m0], m_obj[m0]

        m1 = _finite_mask(ra, dec, m_cat)
        ra, dec, m_cat = ra[m1], dec[m1], m_cat[m1]

        # Rough sky center (used for residuals and SIP check)
        try:
            ra0, dec0 = float(wcs_init.wcs.crval[0]), float(wcs_init.wcs.crval[1])
            if not (np.isfinite(ra0) and np.isfinite(dec0)):
                raise ValueError
        except Exception:
            ra0, dec0 = float(np.nanmedian(ra)), float(np.nanmedian(dec))

        # Pixel scale for radius conversion (arcsec/pixel)
        try:
            pscales = wcs_init.proj_plane_pixel_scales()
            # proj_plane_pixel_scales() may return Quantity with units
            if hasattr(pscales[0], 'to_value'):
                pixel_scale_deg = float(np.mean([s.to_value(u.deg) for s in pscales]))
            else:
                pixel_scale_deg = float(np.mean(pscales))
        except Exception:
            pixel_scale_deg = None
        if pixel_scale_deg is None or pixel_scale_deg <= 0 or not np.isfinite(pixel_scale_deg):
            raise RuntimeError("Could not determine pixel scale from WCS.")
        pixel_scale_arcsec = pixel_scale_deg * 3600.0

        # --- Adaptive source count retry loop (SCAMP-inspired) ---
        # Try pattern matching with increasing source counts. On each retry,
        # double n_det/n_ref to bring in more sources for matching.

        # Pre-sort by brightness (avoids re-sorting each retry)
        obj_order = np.argsort(m_obj)  # lower mag = brighter
        cat_order = np.argsort(m_cat)

        # Project full catalog to pixel space once (slice per retry)
        all_ref_xy = np.column_stack(wcs_init.all_world2pix(ra, dec, 0))
        all_ref_finite = np.all(np.isfinite(all_ref_xy), axis=1)

        # Local config copy so auto_match_resolution doesn't mutate self.cfg
        cfg = dataclasses.replace(self.cfg)

        n_retries = max(0, cfg.adaptive_n_retry)
        last_error = None
        pairs = None

        for _attempt in range(n_retries + 1):
            scale_factor = 2 ** _attempt
            cur_n_det = min(cfg.n_det * scale_factor, len(x))
            cur_n_ref = min(cfg.n_ref * scale_factor, len(ra))

            # Select bright subsets (pre-sorted, just slice)
            obj_sub_idx = obj_order[:cur_n_det]
            cat_sub_idx = cat_order[:cur_n_ref]

            det_xy = np.column_stack([x[obj_sub_idx], y[obj_sub_idx]])
            det_mag = m_obj[obj_sub_idx]

            # Slice pre-projected catalog and filter non-finite
            ref_uv = all_ref_xy[cat_sub_idx]
            ref_mag = m_cat[cat_sub_idx]
            ref_ok = all_ref_finite[cat_sub_idx]
            if not np.all(ref_ok):
                ref_uv = ref_uv[ref_ok]
                ref_mag = ref_mag[ref_ok]
                cat_sub_idx = cat_sub_idx[ref_ok]

            if len(det_xy) < 8 or len(ref_uv) < 8:
                last_error = _PatternMatchFailed("Not enough points after selection.")
                continue

            # Auto matching resolution: widen stage radii from source density
            if cfg.auto_match_resolution:
                auto_r_pix = _auto_match_resolution(det_xy, ref_uv)
                if auto_r_pix is not None:
                    auto_r_arcsec = auto_r_pix * pixel_scale_arcsec
                    cfg = dataclasses.replace(
                        cfg,
                        stage1_radius_arcsec=max(cfg.stage1_radius_arcsec, auto_r_arcsec * 2.5),
                        stage2_radius_arcsec=max(cfg.stage2_radius_arcsec, auto_r_arcsec),
                    )

            try:
                pairs, top_hyp, best = self._pattern_match(
                    det_xy, det_mag, ref_uv, ref_mag, wcs_init, pixel_scale_arcsec,
                    cfg=cfg,
                )
            except _PatternMatchFailed as e:
                last_error = e
                if cur_n_det < len(x) or cur_n_ref < len(ra):
                    continue
                else:
                    break

            # Check if we have enough inliers
            if len(pairs) >= cfg.adaptive_min_inliers:
                break
            # Not enough inliers — retry with more sources if possible
            last_error = _PatternMatchFailed(
                f"Only {len(pairs)} inliers (need {cfg.adaptive_min_inliers})")
            if cur_n_det >= len(x) and cur_n_ref >= len(ra):
                break  # can't add more sources

        if pairs is None or len(pairs) < 8:
            raise RuntimeError(str(last_error) if last_error else "Pattern matching failed.")

        # Build match list for final fit
        pairs_arr = np.array(pairs, dtype=np.int32)
        det_indices = pairs_arr[:, 0]
        ref_indices = pairs_arr[:, 1]

        det_match_idx = obj_sub_idx[det_indices]
        ref_match_idx = cat_sub_idx[ref_indices]

        seed_matches = int(len(det_match_idx))
        used_expanded = False
        refine_pool_det = int(len(det_match_idx))
        refine_pool_ref = int(len(ref_match_idx))

        # Optional: expand refinement pool using many more detections/catalog entries
        if self.cfg.refine_use_all:
            det_pool_idx = np.arange(len(x))
            if self.cfg.refine_n_det is not None and self.cfg.refine_n_det > 0:
                det_pool_idx = np.argsort(m_obj)[: min(self.cfg.refine_n_det, len(m_obj))]
            ref_pool_idx = np.arange(len(ra))
            if self.cfg.refine_n_ref is not None and self.cfg.refine_n_ref > 0:
                ref_pool_idx = np.argsort(m_cat)[: min(self.cfg.refine_n_ref, len(m_cat))]

            if len(det_pool_idx) >= 8 and len(ref_pool_idx) >= 8:
                det_xy_pool = np.column_stack([x[det_pool_idx], y[det_pool_idx]])
                # Project expanded catalog to pixel space using initial WCS
                ref_x_all, ref_y_all = wcs_init.all_world2pix(
                    ra[ref_pool_idx], dec[ref_pool_idx], 0
                )
                ref_uv_all = np.column_stack([ref_x_all, ref_y_all])
                # Filter non-finite
                _ref_ok = np.all(np.isfinite(ref_uv_all), axis=1)
                if not np.all(_ref_ok):
                    ref_uv_all = ref_uv_all[_ref_ok]
                    ref_pool_idx = ref_pool_idx[_ref_ok]

                # Use affine from last rematch iteration for better projection
                try:
                    _pa = np.array(pairs, dtype=np.int32)
                    _A = det_xy[_pa[:, 0]]
                    _B = ref_uv[_pa[:, 1]]
                    M_exp, t_exp = estimate_affine_2d(_A, _B)
                    det_uv_pool = apply_affine(det_xy_pool, M_exp, t_exp)
                except Exception:
                    det_uv_pool = apply_similarity(det_xy_pool, best["R"], best["t"], best["s"])

                r_match_arcsec = self.cfg.refine_match_radius_arcsec
                if r_match_arcsec is None:
                    r_match_arcsec = self.cfg.stage2_radius_arcsec
                r_match = float(r_match_arcsec) / pixel_scale_arcsec

                # Mutual nearest-neighbor for expanded pool
                exp_det_ids, exp_ref_ids = _mutual_nearest_neighbor(
                    det_uv_pool, ref_uv_all, r_match
                )

                if len(exp_det_ids) >= 8:
                    det_match_idx = det_pool_idx[exp_det_ids]
                    ref_match_idx = ref_pool_idx[exp_ref_ids]
                    used_expanded = True
                    refine_pool_det = int(len(det_pool_idx))
                    refine_pool_ref = int(len(ref_pool_idx))

        det_x = x[det_match_idx]
        det_y = y[det_match_idx]
        ref_ra = ra[ref_match_idx]
        ref_dec = dec[ref_match_idx]
        ref_m = m_cat[ref_match_idx]
        det_m = m_obj[det_match_idx]
        refine_matches_preclip = int(len(det_match_idx))

        # Minimum match floor (#6): don't clip below this count
        min_matches = max(8, int(refine_matches_preclip * self.cfg.refine_min_match_fraction))

        # --- Progressive sigma-clipping refinement (#6) ---
        sky = SkyCoord(ref_ra * u.deg, ref_dec * u.deg, frame="icrs")
        xy = np.vstack([det_x, det_y])  # (2,N)

        refined_wcs = None
        keep = np.ones(xy.shape[1], dtype=bool)

        # Progressive clipping: interpolate from start to end sigma
        clip_start = self.cfg.refine_clip_sigma_start
        clip_end = self.cfg.refine_clip_sigma
        n_iter = self.cfg.refine_max_iter

        for it in range(n_iter):
            # Linearly interpolate clip threshold from start (loose) to end (tight)
            if n_iter > 1:
                frac = it / (n_iter - 1)
            else:
                frac = 1.0
            clip_sigma = clip_start + (clip_end - clip_start) * frac

            xy_use = xy[:, keep]
            sky_use = sky[keep]

            # Wrapper handles SIP for TAN, PV for ZPN, linear for others
            refined_wcs = fit_wcs_from_points(
                xy_use,
                sky_use,
                projection=wcs_init,
                sip_degree=int(self.cfg.sip_degree),
            )

            # Residuals via spherical distance (projection-independent)
            ra_fit, dec_fit = refined_wcs.all_pix2world(xy_use[0], xy_use[1], 0)
            ref_ra_use = sky_use.ra.deg
            ref_dec_use = sky_use.dec.deg
            cos_dec = np.cos(np.deg2rad(0.5 * (dec_fit + ref_dec_use)))
            dra_deg = (ra_fit - ref_ra_use) * cos_dec
            ddec_deg = dec_fit - ref_dec_use
            dr = np.deg2rad(np.hypot(dra_deg, ddec_deg))

            sig = _robust_sigma(dr)
            if not np.isfinite(sig) or sig <= 0:
                break

            thresh = clip_sigma * sig
            dr_full = np.full(keep.shape[0], np.nan, dtype=np.float64)
            dr_full[keep] = dr
            keep_new = keep & np.isfinite(dr_full) & (dr_full < thresh)

            # Enforce minimum match floor (#6)
            if keep_new.sum() < min_matches:
                # Don't clip further if we'd go below the minimum
                if keep_new.sum() >= 8:
                    keep = keep_new
                break

            if keep_new.sum() == keep.sum():
                keep = keep_new
                break
            keep = keep_new

        if refined_wcs is None:
            raise RuntimeError("WCS refinement failed in fit_wcs_from_points stage.")

        # Output match table
        final_idx = np.nonzero(keep)[0]
        match = Table()
        match["x"] = det_x[final_idx]
        match["y"] = det_y[final_idx]
        match["obj_mag"] = det_m[final_idx]
        match["ra"] = ref_ra[final_idx]
        match["dec"] = ref_dec[final_idx]
        match["cat_mag"] = ref_m[final_idx]

        # Residuals in arcsec using final WCS (spherical, projection-independent)
        ra_fit, dec_fit = refined_wcs.all_pix2world(match["x"], match["y"], 0)
        cos_dec_out = np.cos(np.deg2rad(0.5 * (dec_fit + _as_float_array(match["dec"]))))
        match["du_arcsec"] = (ra_fit - _as_float_array(match["ra"])) * cos_dec_out * 3600.0
        match["dv_arcsec"] = (dec_fit - _as_float_array(match["dec"])) * 3600.0
        match["dr_arcsec"] = np.hypot(match["du_arcsec"], match["dv_arcsec"])

        diagnostics = {
            "rough_center_deg": (float(ra0), float(dec0)),
            "stage1_hypotheses_kept": int(len(top_hyp)),
            "pattern_inliers_stage2": int(best["inliers"]),
            "seed_matches": int(seed_matches),
            "refine_used_expanded": bool(used_expanded),
            "refine_pool_det": int(refine_pool_det),
            "refine_pool_ref": int(refine_pool_ref),
            "refine_matches_preclip": int(refine_matches_preclip),
            "final_matches": int(len(match)),
            "rms_dr_arcsec": float(np.sqrt(np.mean(match["dr_arcsec"] ** 2))) if len(match) else np.nan,
            "mad_dr_arcsec": float(_robust_sigma(_as_float_array(match["dr_arcsec"]))) if len(match) else np.nan,
            "config": self.cfg,
        }

        return refined_wcs, match, diagnostics


def refine_wcs_quadhash(
    obj: Table,
    cat: Table,
    wcs: WCS = None,
    header = None,
    sr: float = 10 / 3600,
    order: int = 2,
    cat_col_ra: str = 'RAJ2000',
    cat_col_dec: str = 'DEJ2000',
    cat_col_mag: str = 'rmag',
    obj_col_mag: str = None,
    obj_col_flux: str = 'flux',
    sn: float = None,
    n_det: int = 150,
    n_ref: int = 600,
    max_quads_tested: int = 8000,
    allow_reflection: bool = True,
    use_wcs_prior: bool = True,
    scale_tolerance: float = 0.30,
    enforce_parity: bool = True,
    refine_use_all: bool = False,
    refine_n_det: Optional[int] = None,
    refine_n_ref: Optional[int] = None,
    refine_match_radius_arcsec: Optional[float] = None,
    get_header: bool = False,
    update: bool = False,
    verbose: bool = False,
) -> WCS:
    """Refine WCS using quad-hash pattern matching algorithm.

    Pure Python implementation of astrometric refinement using quad-hash based
    pattern matching. Provides robust WCS refinement with no external dependencies.

    :param obj: List of objects on the frame that should contain at least `x`, `y` and either `flux` or `mag` columns.
    :param cat: Reference astrometric catalogue with RA, Dec, and magnitude columns
    :param wcs: Initial WCS solution (rough estimate)
    :param header: FITS header containing initial astrometric solution (alternative to wcs parameter)
    :param sr: Matching radius in degrees (used for stage 2 matching)
    :param order: Polynomial order for SIP distortion solution (1-3, default 2)
    :param cat_col_ra: Catalogue column name for Right Ascension (default 'RAJ2000')
    :param cat_col_dec: Catalogue column name for Declination (default 'DEJ2000')
    :param cat_col_mag: Catalogue column name for magnitude (default 'rmag')
    :param obj_col_mag: Object list column name for magnitude (default: auto-detect 'mag')
    :param obj_col_flux: Object list column name for flux (default 'flux', used if mag not present)
    :param sn: If provided, only objects with signal to noise ratio exceeding this value will be used
    :param n_det: Number of brightest detections to use for matching (default 150)
    :param n_ref: Number of brightest catalog stars to use for matching (default 600)
    :param max_quads_tested: Maximum number of detection quads to test (default 8000)
    :param allow_reflection: Allow reflected quad matches (default True)
    :param use_wcs_prior: Use initial WCS to filter hypotheses by scale/parity (default True)
    :param scale_tolerance: Fractional tolerance for scale filtering (default 0.30)
    :param enforce_parity: Enforce parity consistency with initial WCS (default True)
    :param refine_use_all: If True, expand final refinement to a larger pool of objects/catalog entries
    :param refine_n_det: Max number of detections to use in expanded refinement pool (None = all)
    :param refine_n_ref: Max number of catalog stars to use in expanded refinement pool (None = all)
    :param refine_match_radius_arcsec: Matching radius (arcsec) for expanded refinement pool (default: stage2 radius)
    :param get_header: If True, function will return the FITS header object instead of WCS solution
    :param update: If set, the object list will be updated in-place to contain correct `ra` and `dec` sky coordinates
    :param verbose: Whether to show verbose messages during the run. May be either boolean, or a `print`-like function.
    :returns: Refined WCS solution (or FITS header if get_header=True), or None if refinement fails

    .. note::
        This is a pure Python implementation with no external dependencies (only numpy, scipy, astropy).
        It typically achieves sub-arcsecond accuracy but is ~4x slower than SCAMP (~2s vs ~0.5s).
        However, it is 2-7x more accurate than SCAMP, especially for challenging conditions.

    Example:
        >>> from stdpipe.astrometry_quad import refine_wcs_quadhash
        >>> wcs_refined = refine_wcs_quadhash(
        ...     obj, cat, wcs_init,
        ...     sr=10/3600,  # 10 arcsec matching radius
        ...     order=2,     # Quadratic SIP distortion
        ...     sn=5,        # Use only S/N > 5 objects
        ...     verbose=True
        ... )
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    # Get WCS from header if needed
    if wcs is None and header is not None:
        from astropy.io import fits
        wcs = WCS(header)

    if wcs is None or not wcs.is_celestial:
        log("Error: Valid initial WCS required")
        return None

    # Prepare catalog columns
    if cat_col_ra not in cat.colnames or cat_col_dec not in cat.colnames:
        log(f"Error: Catalog must contain '{cat_col_ra}' and '{cat_col_dec}' columns")
        return None

    if cat_col_mag not in cat.colnames:
        log(f"Warning: Catalog magnitude column '{cat_col_mag}' not found, using first magnitude column")
        # Try to find any magnitude column
        mag_cols = [c for c in cat.colnames if 'mag' in c.lower()]
        if mag_cols:
            cat_col_mag = mag_cols[0]
            log(f"  Using '{cat_col_mag}' as magnitude column")
        else:
            log("Error: No magnitude column found in catalog")
            return None

    # Prepare object columns
    if obj_col_mag is None:
        # Auto-detect magnitude column
        if 'mag' in obj.colnames:
            obj_col_mag = 'mag'
        else:
            obj_col_mag = None

    has_mag = obj_col_mag is not None and obj_col_mag in obj.colnames
    has_flux = obj_col_flux in obj.colnames

    if not has_mag and not has_flux:
        log(f"Error: Object list must contain either '{obj_col_mag or 'mag'}' or '{obj_col_flux}' column")
        return None

    # Apply S/N filtering if requested
    obj_filtered = obj
    if sn is not None:
        if 'fluxerr' in obj.colnames or 'flux_err' in obj.colnames:
            err_col = 'fluxerr' if 'fluxerr' in obj.colnames else 'flux_err'
            flux_col = obj_col_flux if has_flux else None

            if flux_col and flux_col in obj.colnames:
                obj_sn = obj[flux_col] / obj[err_col]
                mask = obj_sn > sn
                obj_filtered = obj[mask]
                log(f"S/N filtering: {len(obj_filtered)}/{len(obj)} objects with S/N > {sn}")
            else:
                log(f"Warning: Cannot apply S/N filter - flux column '{flux_col}' not found")
        else:
            log("Warning: Cannot apply S/N filter - no flux error column found")

    # Create standardized tables with required columns
    cat_std = Table()
    cat_std['ra'] = cat[cat_col_ra]
    cat_std['dec'] = cat[cat_col_dec]
    cat_std['mag'] = cat[cat_col_mag]

    obj_std = Table()
    obj_std['x'] = obj_filtered['x']
    obj_std['y'] = obj_filtered['y']

    if has_mag:
        obj_std['mag'] = obj_filtered[obj_col_mag]
    if has_flux:
        obj_std['flux'] = obj_filtered[obj_col_flux]

    # Configure refinement
    log("Starting quad-hash WCS refinement...")
    log(f"  Catalog: {len(cat_std)} stars")
    log(f"  Detections: {len(obj_std)} objects")
    log(f"  Initial WCS center: RA={wcs.wcs.crval[0]:.6f}, Dec={wcs.wcs.crval[1]:.6f}")
    log(f"  Matching radius: {sr * 3600:.1f} arcsec")
    log(f"  SIP order: {order}")

    config = AstrometryConfig(
        n_det=min(n_det, len(obj_std)),
        n_ref=min(n_ref, len(cat_std)),
        neighbor_k=10,
        eps_desc=0.02,
        n_scale_bins=6,
        stage1_n_det=min(60, len(obj_std)),
        stage1_radius_arcsec=max(20.0, sr * 3600 * 2),  # Stage 1: 2x looser
        stage2_radius_arcsec=sr * 3600,  # Stage 2: use requested radius
        top_k_hypotheses=80,
        max_quads_tested=max_quads_tested,
        allow_reflection=allow_reflection,
        use_wcs_prior=use_wcs_prior,
        scale_tolerance=scale_tolerance,
        enforce_parity=enforce_parity,
        sip_degree=max(0, min(5, order)),  # Clamp to 0-5
        refine_clip_sigma=4.0,
        refine_clip_sigma_start=5.0,
        refine_min_match_fraction=0.5,
        refine_max_iter=5,
        refine_rematch_iters=2,
        refine_use_all=refine_use_all,
        refine_n_det=refine_n_det,
        refine_n_ref=refine_n_ref,
        refine_match_radius_arcsec=refine_match_radius_arcsec,
    )

    try:
        solver = QuadHashAstrometry(config=config)
        wcs_refined, match, diagnostics = solver.refine(obj=obj_std, cat=cat_std, wcs_init=wcs)

        log(f"Refinement successful:")
        log(f"  Pattern matches: {diagnostics['pattern_inliers_stage2']}")
        log(f"  Final matches: {diagnostics['final_matches']}")
        log(f"  RMS residual: {diagnostics['rms_dr_arcsec']:.3f} arcsec")
        log(f"  MAD residual: {diagnostics['mad_dr_arcsec']:.3f} arcsec")
        log(f"  Refined WCS center: RA={wcs_refined.wcs.crval[0]:.6f}, Dec={wcs_refined.wcs.crval[1]:.6f}")

        # Update object table with sky coordinates if requested
        if update and wcs_refined is not None:
            ra_obj, dec_obj = wcs_refined.all_pix2world(obj['x'], obj['y'], 0)
            obj['ra'] = ra_obj
            obj['dec'] = dec_obj
            log(f"Updated object table with ra/dec coordinates")

        # Return header if requested
        if get_header and wcs_refined is not None:
            return wcs_refined.to_header(relax=True)

        return wcs_refined

    except Exception as e:
        log(f"Refinement failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None
