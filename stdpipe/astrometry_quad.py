"""
Quad-hash astrometry solver (Python-only) with two practical upgrades:

1) Multi-scale quad pools:
   - quads are partitioned into baseline-length *rank bins* (e.g. 6 bins)
   - matching is done only within the same bin, which reduces ambiguity in crowded fields

2) Two-stage hypothesis scoring:
   - Stage 1: cheap scoring on a small subset of detections (top-N bright)
   - Keep only top-K hypotheses
   - Stage 2: full scoring on all selected detections with one-to-one matches
   - Then final WCS refinement via astropy.wcs.utils.fit_wcs_from_points

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

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points


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

def quantize_desc(desc: np.ndarray, eps: float) -> Tuple[int, int, int, int]:
    q = np.floor(desc / eps + 0.5).astype(np.int64)
    return int(q[0]), int(q[1]), int(q[2]), int(q[3])

def neighbors_bins(key: Tuple[int, int, int, int], r: int = 1) -> Iterable[Tuple[int, int, int, int]]:
    a, b, c, d = key
    for da in range(-r, r + 1):
        for db in range(-r, r + 1):
            for dc in range(-r, r + 1):
                for dd in range(-r, r + 1):
                    yield (a + da, b + db, c + dc, d + dd)

def baseline_rank_bins(lengths: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Assign each length to a *rank* bin based on quantiles.
    This avoids needing absolute scale consistency between pixels and tangent plane.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    n = len(lengths)
    if n == 0:
        return np.zeros(0, dtype=np.int16)
    if n_bins <= 1:
        return np.zeros(n, dtype=np.int16)
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

def build_quad_hash_multiscale(points: np.ndarray, mags: Optional[np.ndarray], k: int, eps: float, n_scale_bins: int
                               ) -> Dict[Tuple[int, Tuple[int, int, int, int]], List[QuadEntry]]:
    """
    Hash key: (scale_bin_id, quantized_descriptor)
    """
    quads = make_local_quads(points, k=k)
    if not quads:
        return {}

    descs = []
    lens = []
    for q in quads:
        desc, L = quad_descriptor(points[list(q)])
        descs.append(desc)
        lens.append(L)
    descs = np.asarray(descs, dtype=np.float64)
    lens = np.asarray(lens, dtype=np.float64)

    ok = np.isfinite(descs).all(axis=1) & np.isfinite(lens) & (lens > 0)
    quads_ok = [quads[i] for i in range(len(quads)) if ok[i]]
    descs_ok = descs[ok]
    lens_ok = lens[ok]

    bins = baseline_rank_bins(lens_ok, n_bins=n_scale_bins)

    H: Dict[Tuple[int, Tuple[int, int, int, int]], List[QuadEntry]] = {}
    for q, desc, b in zip(quads_ok, descs_ok, bins):
        key = (int(b), quantize_desc(desc, eps))
        me = None
        if mags is not None:
            me = tuple(float(m) for m in mags[list(q)])
        H.setdefault(key, []).append(QuadEntry(idxs=q, desc=desc, bin_id=int(b), mags=me))
    return H

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

    # Two-stage scoring
    stage1_n_det: int = 50              # cheap scoring uses only brightest subset of selected detections
    stage1_radius_arcsec: float = 20.0  # loose
    stage2_radius_arcsec: float = 8.0   # tighter
    top_k_hypotheses: int = 60          # keep best hypotheses from stage 1
    max_quads_tested: int = 5000        # max detection quads to try
    max_candidates_per_bucket: int = 60 # cap ref candidates pulled per hash bucket

    # Final fit
    sip_degree: int = 3
    refine_clip_sigma: float = 4.0
    refine_max_iter: int = 5


class QuadHashAstrometry:
    def __init__(self, config: AstrometryConfig = AstrometryConfig()):
        self.cfg = config

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

        # Rough sky center
        try:
            ra0, dec0 = float(wcs_init.wcs.crval[0]), float(wcs_init.wcs.crval[1])
            if not (np.isfinite(ra0) and np.isfinite(dec0)):
                raise ValueError
        except Exception:
            ra0, dec0 = float(np.nanmedian(ra)), float(np.nanmedian(dec))

        # Bright subsets
        obj_sub_idx = _pick_brightest_obj(Table({"x": x, "y": y, "mag": m_obj}), self.cfg.n_det)
        cat_sub_idx = _pick_brightest_cat(Table({"ra": ra, "dec": dec, "mag": m_cat}), self.cfg.n_ref)

        det_xy = np.column_stack([x[obj_sub_idx], y[obj_sub_idx]])
        det_mag = m_obj[obj_sub_idx]

        ref_u, ref_v = tan_project_deg(ra[cat_sub_idx], dec[cat_sub_idx], ra0, dec0)
        ref_uv = np.column_stack([ref_u, ref_v])
        ref_mag = m_cat[cat_sub_idx]

        if len(det_xy) < 8 or len(ref_uv) < 8:
            raise RuntimeError("Not enough points after selection.")

        # Reference multiscale hash
        ref_hash = build_quad_hash_multiscale(
            ref_uv, ref_mag,
            k=self.cfg.neighbor_k,
            eps=self.cfg.eps_desc,
            n_scale_bins=self.cfg.n_scale_bins
        )

        # Detection quads + their baseline bins
        det_quads = make_local_quads(det_xy, k=self.cfg.neighbor_k)
        if not det_quads:
            raise RuntimeError("Not enough detections for quad matching.")

        det_descs = []
        det_lens = []
        for q in det_quads:
            desc, L = quad_descriptor(det_xy[list(q)])
            det_descs.append(desc)
            det_lens.append(L)
        det_descs = np.asarray(det_descs, dtype=np.float64)
        det_lens = np.asarray(det_lens, dtype=np.float64)
        okq = np.isfinite(det_descs).all(axis=1) & np.isfinite(det_lens) & (det_lens > 0)
        det_quads = [det_quads[i] for i in range(len(det_quads)) if okq[i]]
        det_descs = det_descs[okq]
        det_lens = det_lens[okq]
        det_bins = baseline_rank_bins(det_lens, n_bins=self.cfg.n_scale_bins)

        # Trees for scoring (tangent plane)
        ref_tree = cKDTree(ref_uv)

        # Stage radii in radians
        r1 = (self.cfg.stage1_radius_arcsec * u.arcsec).to_value(u.rad)
        r2 = (self.cfg.stage2_radius_arcsec * u.arcsec).to_value(u.rad)

        # Stage1 detection subset: brightest among det_xy itself (already bright), so just take first stage1_n by det_mag
        order_det = np.argsort(det_mag)  # smaller = brighter
        det_stage1_ids = order_det[: min(self.cfg.stage1_n_det, len(order_det))]
        det_xy_stage1 = det_xy[det_stage1_ids]

        # Hypothesis storage: keep top-K by stage1 score
        # Store tuples (score, R, t, s)
        top_hyp: List[Tuple[int, np.ndarray, np.ndarray, float]] = []

        rng = np.random.default_rng(12345)

        q_order = np.arange(len(det_quads))
        rng.shuffle(q_order)
        q_order = q_order[: min(self.cfg.max_quads_tested, len(q_order))]

        def _try_insert(score: int, R: np.ndarray, t: np.ndarray, s: float):
            nonlocal top_hyp
            if len(top_hyp) < self.cfg.top_k_hypotheses:
                top_hyp.append((score, R, t, s))
                return
            # replace worst if better
            worst_i = int(np.argmin([h[0] for h in top_hyp]))
            if score > top_hyp[worst_i][0]:
                top_hyp[worst_i] = (score, R, t, s)

        # --- Stage 1: cheap scoring
        for qi in q_order:
            q = det_quads[qi]
            desc = det_descs[qi]
            bin_id = int(det_bins[qi])

            key0 = (bin_id, quantize_desc(desc, self.cfg.eps_desc))

            # Gather candidates from neighboring descriptor bins (same scale bin)
            candidates: List[QuadEntry] = []
            _, kdesc = key0
            for kdesc2 in neighbors_bins(kdesc, r=self.cfg.bin_neighbor_radius):
                kk = (bin_id, kdesc2)
                if kk in ref_hash:
                    candidates.extend(ref_hash[kk])

            if not candidates:
                continue

            # Optional quad mag signature
            det_sig = mag_signature(det_mag[list(q)]) if self.cfg.use_mag_signature else None

            rng.shuffle(candidates)
            for ce in candidates[: self.cfg.max_candidates_per_bucket]:
                if det_sig is not None and ce.mags is not None:
                    if mag_signature(np.array(ce.mags)) != det_sig:
                        continue

                P = det_xy[list(q)]
                Q = ref_uv[list(ce.idxs)]

                try:
                    R, t, s = estimate_similarity_2d(P, Q, allow_reflection=self.cfg.allow_reflection)
                except Exception:
                    continue

                # Cheap score on stage1 subset
                det_uv1 = apply_similarity(det_xy_stage1, R, t, s)
                dist, nn = ref_tree.query(det_uv1, k=1, distance_upper_bound=r1)
                ok = np.isfinite(dist) & (dist < r1) & (nn < len(ref_uv))
                score = int(np.sum(ok))
                if score <= 0:
                    continue

                _try_insert(score, R, t, s)

        if not top_hyp:
            raise RuntimeError("Pattern matching failed at stage 1. Try increasing n_det/n_ref, eps_desc, or stage1_radius_arcsec.")

        # --- Stage 2: full scoring of top hypotheses
        # Sort hypotheses by stage1 score descending
        top_hyp.sort(key=lambda x: x[0], reverse=True)

        best = {"inliers": -1, "R": None, "t": None, "s": None, "pairs": None}

        for (score1, R, t, s) in top_hyp:
            det_uv = apply_similarity(det_xy, R, t, s)
            dist, nn = ref_tree.query(det_uv, k=1, distance_upper_bound=r2)
            ok = np.isfinite(dist) & (dist < r2) & (nn < len(ref_uv))
            if not np.any(ok):
                continue

            det_ids = np.nonzero(ok)[0]
            ref_ids = nn[ok].astype(int)
            order = np.argsort(dist[ok])
            det_ids = det_ids[order]
            ref_ids = ref_ids[order]

            # One-to-one greedy
            seen = set()
            pairs = []
            for di, rj in zip(det_ids, ref_ids):
                if rj in seen:
                    continue
                seen.add(rj)
                pairs.append((int(di), int(rj)))

            if len(pairs) > best["inliers"]:
                best = {"inliers": len(pairs), "R": R, "t": t, "s": s, "pairs": pairs}

        if best["inliers"] < 8:
            raise RuntimeError(f"Pattern matching failed at stage 2 (best inliers={best['inliers']}). "
                               f"Try loosening stage2_radius_arcsec or increasing top_k_hypotheses.")

        # Build match list for final fit
        pairs = best["pairs"]
        det_x = det_xy[[p[0] for p in pairs], 0]
        det_y = det_xy[[p[0] for p in pairs], 1]
        ref_ra = ra[cat_sub_idx[[p[1] for p in pairs]]]
        ref_dec = dec[cat_sub_idx[[p[1] for p in pairs]]]
        ref_m = m_cat[cat_sub_idx[[p[1] for p in pairs]]]
        det_m = det_mag[[p[0] for p in pairs]]

        # Final refinement via fit_wcs_from_points + clipping
        sky = SkyCoord(ref_ra * u.deg, ref_dec * u.deg, frame="icrs")
        xy = np.vstack([det_x, det_y])  # (2,N)

        refined_wcs = None
        keep = np.ones(xy.shape[1], dtype=bool)

        for it in range(self.cfg.refine_max_iter):
            xy_use = xy[:, keep]
            sky_use = sky[keep]

            try:
                refined_wcs = fit_wcs_from_points(
                    xy_use,
                    sky_use,
                    projection=wcs_init,
                    sip_degree=int(self.cfg.sip_degree),
                )
            except TypeError:
                # Older astropy versions don't support sip_degree
                refined_wcs = fit_wcs_from_points(
                    xy_use,
                    sky_use,
                    projection=wcs_init,
                )

            # residuals in tangent plane for robust clipping
            ra_fit, dec_fit = refined_wcs.all_pix2world(xy_use[0], xy_use[1], 0)
            u_fit, v_fit = tan_project_deg(ra_fit, dec_fit, ra0, dec0)
            u_ref, v_ref = tan_project_deg(sky_use.ra.deg, sky_use.dec.deg, ra0, dec0)
            dr = np.hypot(u_fit - u_ref, v_fit - v_ref)

            sig = _robust_sigma(dr)
            if not np.isfinite(sig) or sig <= 0:
                break

            thresh = self.cfg.refine_clip_sigma * sig
            dr_full = np.full(keep.shape[0], np.nan, dtype=np.float64)
            dr_full[keep] = dr
            keep_new = keep & np.isfinite(dr_full) & (dr_full < thresh)

            if keep_new.sum() == keep.sum() or keep_new.sum() < 8:
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

        # residuals in arcsec using final WCS
        ra_fit, dec_fit = refined_wcs.all_pix2world(match["x"], match["y"], 0)
        u_fit, v_fit = tan_project_deg(ra_fit, dec_fit, ra0, dec0)
        u_ref, v_ref = tan_project_deg(match["ra"], match["dec"], ra0, dec0)
        du = u_fit - u_ref
        dv = v_fit - v_ref
        match["du_arcsec"] = (du * u.rad).to_value(u.arcsec)
        match["dv_arcsec"] = (dv * u.rad).to_value(u.arcsec)
        match["dr_arcsec"] = np.hypot(match["du_arcsec"], match["dv_arcsec"])

        diagnostics = {
            "rough_center_deg": (float(ra0), float(dec0)),
            "stage1_hypotheses_kept": int(len(top_hyp)),
            "pattern_inliers_stage2": int(best["inliers"]),
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
    :param get_header: If True, function will return the FITS header object instead of WCS solution
    :param update: If set, the object list will be updated in-place to contain correct `ra` and `dec` sky coordinates
    :param verbose: Whether to show verbose messages during the run. May be either boolean, or a `print`-like function.
    :returns: Refined WCS solution (or FITS header if get_header=True), or None if refinement fails

    .. note::
        This is a pure Python implementation with no external dependencies (only numpy, scipy, astropy).
        It typically achieves sub-arcsecond accuracy but is ~4× slower than SCAMP (~2s vs ~0.5s).
        However, it is 2-7× more accurate than SCAMP, especially for challenging conditions.

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
        stage1_radius_arcsec=max(20.0, sr * 3600 * 2),  # Stage 1: 2× looser
        stage2_radius_arcsec=sr * 3600,  # Stage 2: use requested radius
        top_k_hypotheses=80,
        max_quads_tested=max_quads_tested,
        sip_degree=max(0, min(3, order)),  # Clamp to 0-3
        refine_clip_sigma=4.0,
        refine_max_iter=5,
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
