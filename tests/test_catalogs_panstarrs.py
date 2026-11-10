"""Offline regression tests for the direct MAST Pan-STARRS catalogue interface."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import MaskedColumn, Table
from astroquery.exceptions import RemoteServiceError

from stdpipe import catalogs


def _mast_page(ids, *, missing=False, bad_position=False, extra=False):
    """Produce the native columns returned by the Pan-STARRS mean-object API."""
    n = len(ids)
    cat = Table()
    cat["objID"] = np.asarray(ids, dtype=np.int64)
    cat["raMean"] = np.full(n, 210.8)
    cat["decMean"] = np.full(n, 54.35)
    cat["raMeanErr"] = np.full(n, 0.03)
    cat["decMeanErr"] = np.full(n, 0.04)
    for band in "grizy":
        cat[f"{band}MeanPSFMag"] = np.full(n, 17.0)
        cat[f"{band}MeanPSFMagErr"] = np.full(n, 0.02)
    if missing and n >= 3:
        cat["gMeanPSFMag"][0] = -999
        cat["rMeanPSFMag"][1] = np.nan
        cat.replace_column("iMeanPSFMag", MaskedColumn(
            np.full(n, 17.0), mask=[False, False, True] + [False] * (n - 3)))
    if bad_position and n >= 2:
        cat["raMean"][0] = np.nan
        cat["decMean"][1] = -999
    if extra:
        cat["rMeanKronMag"] = np.full(n, 16.9)
    return cat


def _fake_mast(monkeypatch, pages, *, error_page=None):
    calls = []

    class FakeCatalogs:
        def query_region(self, *args, **kwargs):
            calls.append(kwargs)
            if kwargs["page"] == error_page:
                raise RemoteServiceError("MAST unavailable")
            return pages.get(kwargs["page"], _mast_page([]))

    monkeypatch.setattr(catalogs, "CatalogsClass", FakeCatalogs)
    return calls


@pytest.mark.unit
def test_panstarrs_multiple_pages_and_exact_page_boundary(monkeypatch):
    calls = _fake_mast(monkeypatch, {
        1: _mast_page([1, 2]), 2: _mast_page([3, 4]),
    })
    cat = catalogs.get_cat_panstarrs(210.8, 54.35, 0.1, page_size=2,
                                     augment_bands=False)
    assert list(cat["objID"]) == [1, 2, 3, 4]
    assert [call["page"] for call in calls] == [1, 2, 3]
    assert all(call["pagesize"] == 2 and call["sort_by"] == "objID" for call in calls)
    assert all(call["data_release"] == "dr2" and call["table"] == "mean" for call in calls)
    assert cat.meta["release"] == "dr2"
    assert "nDetections" not in calls[0]
    assert cat["e_RAJ2000"].unit == u.arcsec
    assert cat["gmag"].unit == u.mag


@pytest.mark.unit
def test_panstarrs_limit_and_zero_limit(monkeypatch):
    calls = _fake_mast(monkeypatch, {1: _mast_page([1, 2]), 2: _mast_page([3])})
    cat = catalogs.get_cat_panstarrs(210.8, 54.35, 0.1, page_size=2,
                                     limit=3, augment_bands=False)
    assert len(cat) == 3
    assert [call["pagesize"] for call in calls] == [2, 2]
    assert len(calls) == 2

    calls.clear()
    empty = catalogs.get_cat_panstarrs(210.8, 54.35, 0.1, limit=0,
                                       augment_bands=False)
    assert isinstance(empty, Table) and len(empty) == 0
    assert "gmag" in empty.colnames
    assert not calls


@pytest.mark.unit
def test_panstarrs_missing_photometry_and_positions(monkeypatch):
    _fake_mast(monkeypatch, {1: _mast_page([10, 11, 12], missing=True)})
    cat = catalogs.get_cat_panstarrs(210.8, 54.35, 0.1,
                                     augment_bands=False)
    assert np.ma.getmaskarray(cat["gmag"])[0]
    assert np.ma.getmaskarray(cat["rmag"])[1]
    assert np.ma.getmaskarray(cat["imag"])[2]
    assert not np.ma.getmaskarray(cat["zmag"]).any()

    _fake_mast(monkeypatch, {1: _mast_page([10, 11, 12], bad_position=True)})
    cat = catalogs.get_cat_panstarrs(210.8, 54.35, 0.1,
                                     augment_bands=False)
    assert list(cat["objID"]) == [12]


@pytest.mark.unit
def test_panstarrs_native_filters_extra_columns_and_distance(monkeypatch):
    calls = _fake_mast(monkeypatch, {1: _mast_page([1], extra=True)})
    filters = {"rMeanPSFMag": [("lt", 18)], "nDetections": [("gte", 2)]}
    cat = catalogs.get_cat_panstarrs(
        210.8, 54.35, 0.1, filters=filters, extra=["rMeanKronMag"],
        get_distance=True, augment_bands=False)
    assert "rMeanKronMag" in cat.colnames
    assert "_r" in cat.colnames and cat["_r"].unit == u.deg
    assert all(calls[0][key] == value for key, value in filters.items())
    assert "rMeanKronMag" in calls[0]["columns"]


@pytest.mark.unit
def test_panstarrs_failed_later_page_never_returns_partial_catalog(monkeypatch):
    _fake_mast(monkeypatch, {1: _mast_page([1, 2])}, error_page=2)
    assert catalogs.get_cat_panstarrs(210.8, 54.35, 0.1, page_size=2,
                                      augment_bands=False) is None


@pytest.mark.unit
def test_panstarrs_missing_required_response_column(monkeypatch):
    page = _mast_page([1])
    page.remove_column("gMeanPSFMag")
    _fake_mast(monkeypatch, {1: page})
    assert catalogs.get_cat_panstarrs(210.8, 54.35, 0.1,
                                      augment_bands=False) is None


@pytest.mark.unit
def test_panstarrs_augmentation(monkeypatch):
    _fake_mast(monkeypatch, {1: _mast_page([1])})
    calls = []

    def fake_augment(cat, catalog=None, verbose=False):
        calls.append(catalog)
        cat["Vmag"] = cat["gmag"]
        return cat

    monkeypatch.setattr(catalogs, "augment_cat_bands", fake_augment)
    cat = catalogs.get_cat_panstarrs(210.8, 54.35, 0.1)
    assert calls == ["ps1"]
    assert "Vmag" in cat.colnames


@pytest.mark.unit
@pytest.mark.parametrize("kwargs", [
    {"sr0": 0}, {"sr0": 1.1}, {"ra0": float("nan")},
    {"limit": -2}, {"page_size": 0},
    {"filters": {"page": 2}}, {"filters": {"COLUMNS": ["objID"]}},
])
def test_panstarrs_invalid_parameters(kwargs):
    params = {"ra0": 210.8, "dec0": 54.35, "sr0": 0.1}
    params.update(kwargs)
    with pytest.raises(ValueError):
        catalogs.get_cat_panstarrs(**params)
