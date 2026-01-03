
import sys
import numpy as np
from astropy.stats import mad_std

from scipy.spatial import cKDTree

from . import astrometry


class LCs:
    """
    Container for light-curve data vectors with spatial clustering utilities.

    Stores user-provided per-detection vectors (e.g., ra/dec/flux/time) and
    groups detections into spatial clusters using a KDTree radius search.
    Clustering returns per-cluster centroids and member indices in `self.lcs`.

    Notes
    -----
    - `add()` broadcasts scalars to the length of the longest input vector.
    - `cluster()` refines centroids and can call an `analyze(self, ids)` callback
      per cluster.
    - Coordinate jitter is applied when building the KDTree to avoid degeneracy
      from repeated positions.
    - Clustering results are stored in `self.lcs` with keys:
      - `x`, `y`, `z`: centroid unit-vector coordinates.
      - `ra`, `dec`: centroid sky coordinates in degrees.
      - `N`: number of points per cluster.
      - `ids`: list of index arrays for member points in the container.
      - `kd`: KDTree built from centroid vectors for fast queries.
    """
    def __init__(self):
        # Storage for user-supplied data vectors
        self._params = {}

        self.lcs = None
        self.kd = None

    def __getattr__(self, name):
        # Allows direct access to stored data vectors
        if name in self._params:
            return self._params.get(name)

        else:
            raise AttributeError

    def __dir__(self):
        # For auto-completion of stored data vectors names
        return (
            list(self.__dict__.keys())
            + list(self.__class__.__dict__.keys())
            + list(self._params.keys())
        )

    def add(self, **kwargs):
        """
        Add per-detection vectors to the container.

        Each keyword defines a stored vector. Scalars are broadcast to the length
        of the longest input vector, and missing values are filled with None.
        This method may be called repeatedly to append new chunks of measurements
        (e.g., per-image batches) to the existing vectors.

        Examples
        --------
        >>> lcs = LCs()
        >>> lcs.add(ra=[1, 2], dec=[3, 4], flux=10.0)
        """
        def extend(col, val, length):
            if (
                val is not None
                and hasattr(val, "__len__")
                and not isinstance(val, str)
                and len(val) == length
            ):
                col.extend(val)
            elif val is not None:
                col.extend(np.repeat(val, length))
            else:
                col.extend(np.repeat(None, length))

        # Estimate vector length from input data - for now, just as a max length of individual values
        length = 0
        for _ in kwargs:
            if hasattr(kwargs[_], '__len__') and not isinstance(kwargs[_], str):
                length = max(length, len(kwargs[_]))

        for key in kwargs:
            if key not in self._params:
                self._params[key] = []

            extend(self._params[key], kwargs[key], length)

    def cluster(
        self,
        sr=1 / 3600,
        min_length=None,
        col_ra='ra',
        col_dec='dec',
        verbose=True,
        analyze=None,
        N=1000,
        method='auto',
        unionfind_threshold=5000,
        max_refine_iter=1,
    ):
        """
        Spatially cluster the data vectors using ra/dec values stored in `col_ra` and `col_dec`.

        Parameters
        ----------
        sr : float, optional
            Clustering radius in degrees.
        min_length : int or None, optional
            Minimum number of points required to keep a cluster.
        col_ra : str, optional
            Name of the RA column in stored vectors.
        col_dec : str, optional
            Name of the Dec column in stored vectors.
        verbose : bool or callable, optional
            Logging control, can be a print-like function.
        analyze : callable or None, optional
            Optional callback `analyze(self, ids)` called per accepted cluster.
            Any returned mapping entries are appended into `self.lcs` under their
            respective keys (one entry per cluster).
        N : int, optional
            Progress update interval in points.
        method : {'auto', 'unionfind'}, optional
            Use union-find prepass to limit scan to connected components.
        unionfind_threshold : int, optional
            Minimum number of points for enabling union-find when method='auto'.
        max_refine_iter : int, optional
            Maximum number of centroid refinement iterations (default 1).
        """

        log = (
            (verbose if callable(verbose) else print)
            if verbose
            else lambda *args, **kwargs: None
        )

        if min_length is None:
            min_length = 0

        sr0 = np.deg2rad(sr)

        if type(self._params[col_ra]) is not np.ndarray:
            log('Converting arrays')

            for name in self._params:
                self._params[name] = np.array(self._params[name])

            self._xarr, self._yarr, self._zarr = astrometry.radectoxyz(
                self._params[col_ra], self._params[col_dec]
            )
            # Add some additional jitter to coordinates, or KDTree may hang on repeating positions
            self._xarr = np.random.normal(self._xarr, 0.01 / 206265)
            self._yarr = np.random.normal(self._yarr, 0.01 / 206265)
            self._zarr = np.random.normal(self._zarr, 0.01 / 206265)
            self.kd = cKDTree(np.array([self._xarr, self._yarr, self._zarr]).T)

        def refine_pos(x, y, z):
            """Returns mean position for a list of individual positions"""
            x1, y1, z1 = [np.mean(_) for _ in [x, y, z]]

            # Normalize back to unit sphere
            r = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
            x1, y1, z1 = [_ / r for _ in [x1, y1, z1]]

            return x1, y1, z1

        xarr = self._xarr
        yarr = self._yarr
        zarr = self._zarr
        kd = self.kd

        vmask = np.zeros_like(self._params[col_ra], bool)

        self.lcs = {'x': [], 'y': [], 'z': [], 'N': [], 'ids': []}
        lcs_x = self.lcs['x']
        lcs_y = self.lcs['y']
        lcs_z = self.lcs['z']
        lcs_n = self.lcs['N']
        lcs_ids = self.lcs['ids']

        log(
            'Starting spatial clustering of %d points with %.1f arcsec radius'
            % (len(vmask), sr * 3600)
        )

        use_unionfind = method == 'unionfind'
        if method == 'auto':
            use_unionfind = len(vmask) >= unionfind_threshold

        def process_seed(i):
            # Select points around seed position
            ids = kd.query_ball_point([xarr[i], yarr[i], zarr[i]], sr0)

            if len(ids) < min_length:
                vmask[ids] = True
                return

            x1, y1, z1 = refine_pos(xarr[ids], yarr[ids], zarr[ids])
            ids = kd.query_ball_point([x1, y1, z1], sr0)

            for _ in range(max_refine_iter - 1):
                x2, y2, z2 = refine_pos(xarr[ids], yarr[ids], zarr[ids])
                ids2 = kd.query_ball_point([x2, y2, z2], sr0)
                if len(ids2) == len(ids) and np.all(np.in1d(ids2, ids)):
                    x1, y1, z1 = x2, y2, z2
                    break
                ids = ids2
                x1, y1, z1 = x2, y2, z2

            vmask[ids] = True  # Mask all points around mean position

            if len(ids) >= min_length:
                # Actual processing of points
                lcs_x.append(x1)
                lcs_y.append(y1)
                lcs_z.append(z1)
                lcs_n.append(len(ids))
                lcs_ids.append(ids)

                if analyze is not None and callable(analyze):
                    ares = analyze(self, ids)

                    for _, __ in ares.items():
                        if _ not in self.lcs:
                            self.lcs[_] = []
                        self.lcs[_].append(__)

        if use_unionfind:
            pairs = kd.query_pairs(sr0)
            parent = np.arange(len(vmask))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            def union(i, j):
                ri = find(i)
                rj = find(j)
                if ri != rj:
                    parent[rj] = ri

            for i, j in pairs:
                union(i, j)

            comps = {}
            for i in range(len(vmask)):
                root = find(i)
                comps.setdefault(root, []).append(i)

            comp_list = list(comps.values())
        else:
            comp_list = [np.arange(len(vmask))]

        for comp in comp_list:
            for i in comp:
                if not vmask[i]:
                    process_seed(i)

                if i % N == 0 and verbose == True:
                    sys.stdout.write("\r %d points - %d lcs" % (i, len(self.lcs['x'])))
                    sys.stdout.flush()

        if verbose == True:
            sys.stdout.write("\n")
            sys.stdout.flush()

        for _ in self.lcs.keys():
            if isinstance(self.lcs[_], list) and _ not in ['ids']:
                self.lcs[_] = np.array(self.lcs[_])

        self.lcs['ra'], self.lcs['dec'] = astrometry.xyztoradec(
            [self.lcs['x'], self.lcs['y'], self.lcs['z']]
        )
        self.lcs['kd'] = cKDTree(
            np.array([self.lcs['x'], self.lcs['y'], self.lcs['z']]).T
        )

        log('%d spatial clusters isolated' % len(self.lcs['ra']))
