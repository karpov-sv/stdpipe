import numpy as np

from stdpipe import astrometry
from stdpipe.lcs import LCs


def _build_sample_lcs():
    # Two tight clusters separated by ~36 arcsec in RA
    c1_ra = 10.0 + np.array([0.0, 0.1, -0.1]) / 3600.0
    c1_dec = 10.0 + np.array([0.0, 0.05, -0.05]) / 3600.0
    c2_ra = 10.01 + np.array([0.0, 0.05]) / 3600.0
    c2_dec = 10.0 + np.array([0.0, -0.05]) / 3600.0

    ra = np.concatenate([c1_ra, c2_ra])
    dec = np.concatenate([c1_dec, c2_dec])
    flux = np.array([10.0, 11.0, 9.0, 20.0, 21.0])

    lcs = LCs()
    lcs.add(ra=ra, dec=dec, flux=flux)
    return lcs


def test_cluster_groups_points_and_analyze():
    np.random.seed(0)
    lcs = _build_sample_lcs()

    def analyze(obj, ids):
        return {'mean_flux': np.mean(obj.flux[ids])}

    lcs.cluster(sr=1 / 3600, min_length=2, verbose=False, analyze=analyze)

    assert len(lcs.lcs['ra']) == 2
    assert sorted(lcs.lcs['N'].tolist()) == [2, 3]
    assert 'mean_flux' in lcs.lcs
    assert len(lcs.lcs['mean_flux']) == 2


def test_cluster_respects_min_length():
    np.random.seed(0)
    lcs = _build_sample_lcs()

    lcs.cluster(sr=1 / 3600, min_length=4, verbose=False)

    assert len(lcs.lcs['ra']) == 0


def test_cluster_merges_when_radius_large():
    np.random.seed(0)
    lcs = _build_sample_lcs()

    lcs.cluster(sr=60 / 3600, min_length=1, verbose=False)

    assert len(lcs.lcs['ra']) == 1
    assert lcs.lcs['N'][0] == 5


def test_cluster_separates_bridged_overlapping_groups():
    sr = 1 / 3600
    ra0 = 0.0
    dec0 = 0.0
    sep = 1.8 / 3600
    bridge = 0.9 / 3600

    c1_ra = ra0 + np.array([-0.1, 0.0, 0.1]) / 3600
    c2_ra = ra0 + sep + np.array([-0.1, 0.0, 0.1]) / 3600
    c1_dec = dec0 + np.array([-0.05, 0.0, 0.05]) / 3600
    c2_dec = dec0 + np.array([0.05, 0.0, -0.05]) / 3600

    ra = np.concatenate([c1_ra, c2_ra, [ra0 + bridge]])
    dec = np.concatenate([c1_dec, c2_dec, [dec0]])

    np.random.seed(0)
    lcs = LCs()
    lcs.add(ra=ra, dec=dec)
    lcs.cluster(sr=sr, min_length=2, verbose=False)

    assert len(lcs.lcs['ra']) == 2
    # The bridge point falls within sr of both group centroids and ends up
    # claimed by each cluster.
    assert sorted(lcs.lcs['N'].tolist()) == [4, 4]


def test_cluster_gaussian_separation():
    sr = 2.0 / 3600
    sigma = 0.5 * sr
    n_per = 200
    ra0, dec0 = 100.0, 20.0
    sep = 2.0 * sr

    rng = np.random.default_rng(0)
    ra1 = ra0 + rng.normal(scale=sigma, size=n_per)
    dec1 = dec0 + rng.normal(scale=sigma, size=n_per)
    ra2 = ra0 + sep + rng.normal(scale=sigma, size=n_per)
    dec2 = dec0 + rng.normal(scale=sigma, size=n_per)

    ra = np.concatenate([ra1, ra2])
    dec = np.concatenate([dec1, dec2])

    np.random.seed(0)
    lcs = LCs()
    lcs.add(ra=ra, dec=dec)
    lcs.cluster(sr=sr, min_length=20, verbose=False)

    assert len(lcs.lcs['ra']) >= 2

    order = np.argsort(lcs.lcs['N'])[::-1]
    ra_centers = lcs.lcs['ra'][order[:2]]
    dec_centers = lcs.lcs['dec'][order[:2]]
    sep_meas = astrometry.spherical_distance(
        ra_centers[0], dec_centers[0], ra_centers[1], dec_centers[1]
    )
    assert sep_meas > 1.5 * sr
    assert sep_meas < 2.5 * sr
