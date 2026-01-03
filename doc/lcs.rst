Lightcurves
===========


Lightcurve clustering
----------------------

We have minimal support for costructing the light curves from a sets of per-frame measurements of many objects, by applying simple spatial clusering to their positions. It is implemented in the :class:`stdpipe.lcs.LCs` container that stores per-detection vectors (e.g. RA/Dec/flux) and can group them into spatial clusters using a KDTree radius search. Results are stored in ``self.lcs`` with centroid coordinates and member indices.

.. code-block:: python

   lcs = LCs()
   lcs.add(ra=ra, dec=dec, flux=flux, time=time1)
   # Can be called repeatedly, e.g. to append per-image chunks
   lcs.add(ra=ra2, dec=dec2, flux=flux2, time=time2)
   lcs.cluster(sr=1/3600, min_length=3)

   # Cluster centroids (degrees) and sizes
   print(lcs.lcs['ra'], lcs.lcs['dec'], lcs.lcs['N'])

   # Show the lightcurve for first object
   ids = lcs.lcs['ids'][0]
   plt.plot(lcs.time[ids], lcs.flux[ids], 'o--')


The clustering results include:

- ``x``, ``y``, ``z``: centroid unit-vector coordinates
- ``ra``, ``dec``: centroid sky coordinates in degrees
- ``N``: number of points per cluster
- ``ids``: list of index arrays for member points
- ``kd``: KDTree built from centroid vectors

You can pass an ``analyze=`` callback to compute per-cluster diagnostics. The callable
should return a mapping; each value is appended to ``self.lcs`` under its key.

.. code-block:: python

   def analyze(lcs, ids):
       return {'mean_flux': np.mean(lcs.flux[ids])}

   lcs.cluster(sr=1/3600, analyze=analyze)
   print(lcs.lcs['mean_flux'])

Key parameters include ``sr`` (radius in degrees), ``min_length``, and ``method`` to
control the union-find prepass for large datasets.

.. autoclass:: stdpipe.lcs.LCs
   :noindex:
   :members:
