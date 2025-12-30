
import sys
import numpy as np
from astropy.stats import mad_std

from scipy.spatial import cKDTree

from . import astrometry


class LCs:
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
    ):
        """
        Spatially cluster the data vectors using ra/dec values stored in `col_ra` and `col_dec` vectors.
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

        vmask = np.zeros_like(self._params[col_ra], bool)

        self.lcs = {'x': [], 'y': [], 'z': [], 'N': [], 'ids': []}

        log(
            'Starting spatial clustering of %d points with %.1f arcsec radius'
            % (len(vmask), sr * 3600)
        )

        for i in range(len(vmask)):
            if not vmask[i]:
                # Select points around seed position
                ids = self.kd.query_ball_point(
                    [self._xarr[i], self._yarr[i], self._zarr[i]], sr0
                )

                if len(ids) < min_length:
                    vmask[
                        ids
                    ] = (
                        True
                    )  # Should we really mask all points close to seed position here?..
                else:
                    x1, y1, z1 = refine_pos(
                        self._xarr[ids], self._yarr[ids], self._zarr[ids]
                    )
                    ids = self.kd.query_ball_point([x1, y1, z1], sr0)

                    # Select points around mean position
                    ids = self.kd.query_ball_point([x1, y1, z1], sr0)
                    vmask[ids] = True  # Mask all points around mean position

                    if len(ids) >= min_length:
                        # Actual processing of points

                        self.lcs['x'].append(x1)
                        self.lcs['y'].append(y1)
                        self.lcs['z'].append(z1)
                        self.lcs['N'].append(len(ids))
                        self.lcs['ids'].append(ids)

                        if analyze is not None and callable(analyze):
                            ares = analyze(self, ids)

                            for _, __ in ares.items():
                                if _ not in self.lcs:
                                    self.lcs[_] = []
                                self.lcs[_].append(__)

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
