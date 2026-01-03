import numpy as np

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
