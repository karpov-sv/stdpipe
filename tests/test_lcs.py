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


def test_cluster_unionfind_matches_iterative_with_overlap():
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
    lcs_iter = LCs()
    lcs_iter.add(ra=ra, dec=dec)
    lcs_iter.cluster(
        sr=sr, min_length=2, verbose=False, method='auto', unionfind_threshold=10**9
    )

    np.random.seed(0)
    lcs_uf = LCs()
    lcs_uf.add(ra=ra, dec=dec)
    lcs_uf.cluster(sr=sr, min_length=2, verbose=False, method='unionfind')

    assert len(lcs_iter.lcs['ra']) == len(lcs_uf.lcs['ra']) == 2
    assert sorted(lcs_iter.lcs['N'].tolist()) == sorted(lcs_uf.lcs['N'].tolist())

    order_iter = np.argsort(lcs_iter.lcs['ra'])
    order_uf = np.argsort(lcs_uf.lcs['ra'])
    assert np.allclose(lcs_iter.lcs['ra'][order_iter], lcs_uf.lcs['ra'][order_uf], atol=1e-6)
    assert np.allclose(lcs_iter.lcs['dec'][order_iter], lcs_uf.lcs['dec'][order_uf], atol=1e-6)


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
    lcs_iter = LCs()
    lcs_iter.add(ra=ra, dec=dec)
    lcs_iter.cluster(
        sr=sr, min_length=20, verbose=False, method='auto', unionfind_threshold=10**9
    )

    lcs_uf = LCs()
    lcs_uf.add(ra=ra, dec=dec)
    lcs_uf.cluster(sr=sr, min_length=20, verbose=False, method='unionfind')

    assert len(lcs_iter.lcs['ra']) >= 2
    assert len(lcs_uf.lcs['ra']) >= 2

    order = np.argsort(lcs_iter.lcs['N'])[::-1]
    ra_centers = lcs_iter.lcs['ra'][order[:2]]
    dec_centers = lcs_iter.lcs['dec'][order[:2]]
    sep_meas = astrometry.spherical_distance(
        ra_centers[0], dec_centers[0], ra_centers[1], dec_centers[1]
    )
    assert sep_meas > 1.5 * sr
    assert sep_meas < 2.5 * sr

    order_uf = np.argsort(lcs_uf.lcs['N'])[::-1]
    ra_centers_uf = lcs_uf.lcs['ra'][order_uf[:2]]
    dec_centers_uf = lcs_uf.lcs['dec'][order_uf[:2]]
    sep_meas_uf = astrometry.spherical_distance(
        ra_centers_uf[0], dec_centers_uf[0], ra_centers_uf[1], dec_centers_uf[1]
    )
    assert sep_meas_uf > 1.5 * sr
    assert sep_meas_uf < 2.5 * sr
