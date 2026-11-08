"""
Unit tests for stdpipe.catalogs module.

Tests catalog querying and manipulation utilities.
"""

import pytest
import numpy as np
import requests
from astropy.table import Table
from astroquery.exceptions import TableParseError, EmptyResponseError

from stdpipe import catalogs


class TestCatalogDefinitions:
    """Test catalog configuration and definitions."""

    @pytest.mark.unit
    def test_catalogs_dict_exists(self):
        """Test that catalogs dictionary exists and is populated."""
        assert hasattr(catalogs, 'catalogs')
        assert isinstance(catalogs.catalogs, dict)
        assert len(catalogs.catalogs) > 0

    @pytest.mark.unit
    def test_catalog_ps1_definition(self):
        """Test Pan-STARRS catalog definition."""
        assert 'ps1' in catalogs.catalogs
        ps1 = catalogs.catalogs['ps1']

        assert 'vizier' in ps1
        assert 'name' in ps1
        assert ps1['name'] == 'PanSTARRS DR1'

    @pytest.mark.unit
    def test_catalog_gaia_definitions(self):
        """Test that various Gaia catalogs are defined."""
        gaia_catalogs = ['gaiadr2', 'gaiaedr3', 'gaiadr3syn']

        for cat in gaia_catalogs:
            assert cat in catalogs.catalogs
            assert 'vizier' in catalogs.catalogs[cat]
            assert 'Gaia' in catalogs.catalogs[cat]['name']

    @pytest.mark.unit
    def test_all_catalogs_have_required_fields(self):
        """Test that all catalog definitions have required fields."""
        for cat_name, cat_def in catalogs.catalogs.items():
            assert 'vizier' in cat_def, f"Catalog {cat_name} missing 'vizier'"
            assert 'name' in cat_def, f"Catalog {cat_name} missing 'name'"


class TestCatalogQuerying:
    """Test catalog query utilities (network-dependent)."""

    @pytest.mark.unit
    @pytest.mark.requires_network
    @pytest.mark.slow
    def test_get_cat_vizier_basic(self):
        """Test basic Vizier catalog query."""
        # Query a small region
        ra0 = 180.0
        dec0 = 45.0
        sr0 = 0.1  # 0.1 degree radius

        # Query Pan-STARRS
        cat = catalogs.get_cat_vizier(
            ra0, dec0, sr0,
            catalog='ps1',
            limit=10,
            verbose=False
        )

        # Should return a table
        assert isinstance(cat, Table)

        # Should have some standard columns
        # (exact column names may vary)
        assert len(cat.colnames) > 0

    @pytest.mark.unit
    @pytest.mark.requires_network
    @pytest.mark.slow
    def test_get_cat_vizier_with_filters(self):
        """Test Vizier query with column filters."""
        pytest.skip("Network-dependent test - implement as needed")

    @pytest.mark.unit
    @pytest.mark.requires_network
    @pytest.mark.slow
    def test_get_cat_vizier_different_catalogs(self):
        """Test querying different catalogs."""
        pytest.skip("Network-dependent test - implement as needed")

    @pytest.mark.unit
    def test_get_cat_vizier_invalid_catalog(self):
        """Test behavior with invalid catalog name."""
        # Should either raise error or return empty result
        # depending on implementation
        pytest.skip("Need to check error handling behavior")


class TestCatalogProcessing:
    """Test catalog processing and augmentation utilities."""

    @pytest.mark.unit
    def test_catalog_magnitude_conversions(self):
        """Test magnitude conversion utilities if they exist."""
        # The catalogs module augments some catalogs with
        # converted magnitudes (e.g., PS1 to Johnson-Cousins)
        # This would test those conversion functions
        pytest.skip("Magnitude conversion tests - implement based on actual functions")

    @pytest.mark.unit
    def test_catalog_coordinate_matching(self):
        """Test catalog cross-matching utilities."""
        # If there are cross-matching utilities in the module
        pytest.skip("Cross-matching tests - implement based on actual functions")


# ============================================================================
# Mock tests (don't require network)
# ============================================================================

class TestCatalogUtilitiesMocked:
    """Test catalog utilities with mocked data."""

    @pytest.mark.unit
    def test_catalog_result_processing(self):
        """Test processing of catalog query results."""
        # Create a mock catalog result
        mock_cat = Table()
        mock_cat['RAJ2000'] = [180.0, 180.1, 180.2]
        mock_cat['DEJ2000'] = [45.0, 45.1, 45.2]
        mock_cat['gmag'] = [15.0, 16.0, 14.5]
        mock_cat['rmag'] = [14.5, 15.5, 14.0]

        # Test any processing functions that might exist
        # (This is a placeholder - implement based on actual functions)
        assert len(mock_cat) == 3
        assert 'RAJ2000' in mock_cat.colnames


class TestCatalogAugmentation:
    """Test catalog augmentation functionality."""

    @pytest.mark.unit
    def test_detect_catalog_type_ps1(self):
        """Test auto-detection of Pan-STARRS catalogs."""
        cat = Table()
        cat['gmag'] = [15.0]
        cat['rmag'] = [14.5]
        cat['imag'] = [14.0]
        cat['zmag'] = [13.8]
        cat['ymag'] = [13.7]

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'ps1'

    @pytest.mark.unit
    def test_detect_catalog_type_gaiadr2(self):
        """Test auto-detection of Gaia DR2 catalogs."""
        cat = Table()
        cat['Gmag'] = [15.0]
        cat['BPmag'] = [15.5]
        cat['RPmag'] = [14.5]

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'gaiadr2'

    @pytest.mark.unit
    def test_detect_catalog_type_skymapper(self):
        """Test auto-detection of SkyMapper catalogs."""
        cat = Table()
        cat['gPSF'] = [15.0]
        cat['rPSF'] = [14.5]
        cat['iPSF'] = [14.0]
        cat['zPSF'] = [13.8]

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'skymapper'

    @pytest.mark.unit
    def test_detect_catalog_type_apass(self):
        """Test auto-detection of APASS catalogs."""
        cat = Table()
        cat['B_mag'] = [15.0]
        cat['V_mag'] = [14.5]
        cat['g_mag'] = [14.8]
        cat['r_mag'] = [14.3]
        cat['i_mag'] = [14.0]

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'apass'

    @pytest.mark.unit
    def test_detect_catalog_type_gaiadr3syn(self):
        """Test auto-detection of Gaia DR3 Synthetic catalogs."""
        cat = Table()
        cat['Fu'] = [1e-10]
        cat['Fg'] = [1e-9]
        cat['Fr'] = [1e-9]
        cat['Fi'] = [1e-9]
        cat['Fz'] = [1e-9]

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'gaiadr3syn'

    @pytest.mark.unit
    def test_detect_catalog_type_sdss(self):
        """Test auto-detection of SDSS catalogs."""
        cat = Table()
        cat['umag'] = [16.0]
        cat['gmag'] = [15.0]
        cat['rmag'] = [14.5]
        cat['imag'] = [14.0]
        cat['zmag'] = [13.8]
        # No ymag - this distinguishes from PS1

        detected = catalogs._detect_catalog_type(cat)
        assert detected == 'sdss'

    @pytest.mark.unit
    def test_detect_catalog_type_unknown(self):
        """Test that unknown catalogs return None."""
        cat = Table()
        cat['random_col1'] = [1.0]
        cat['random_col2'] = [2.0]

        detected = catalogs._detect_catalog_type(cat)
        assert detected is None

    @pytest.mark.unit
    def test_augment_ps1_basic(self):
        """Test PS1 catalog augmentation adds Johnson-Cousins magnitudes."""
        cat = Table()
        cat['gmag'] = np.array([15.0, 16.0])
        cat['rmag'] = np.array([14.5, 15.5])
        cat['imag'] = np.array([14.0, 15.0])
        cat['zmag'] = np.array([13.8, 14.8])
        cat['ymag'] = np.array([13.7, 14.7])
        cat['e_gmag'] = np.array([0.01, 0.02])
        cat['e_rmag'] = np.array([0.01, 0.02])
        cat['e_imag'] = np.array([0.01, 0.02])

        result = catalogs.augment_cat_bands(cat, catalog='ps1', verbose=False)

        # Should have added Johnson-Cousins magnitudes
        assert 'Bmag' in result.colnames
        assert 'Vmag' in result.colnames
        assert 'Rmag' in result.colnames
        assert 'Imag' in result.colnames

        # Should have added aliases
        assert 'B' in result.colnames
        assert 'V' in result.colnames
        assert 'R' in result.colnames
        assert 'I' in result.colnames

        # Should have added SDSS conversions
        assert 'g_SDSS' in result.colnames
        assert 'r_SDSS' in result.colnames

    @pytest.mark.unit
    def test_augment_gaiadr2(self):
        """Test Gaia DR2 catalog augmentation."""
        cat = Table()
        cat['Gmag'] = np.array([15.0, 16.0])
        cat['BPmag'] = np.array([15.5, 16.5])
        cat['RPmag'] = np.array([14.5, 15.5])
        cat['E_BR_RP_'] = np.array([1.2, 1.3])  # Column name as used in code
        cat['e_Gmag'] = np.array([0.01, 0.02])

        result = catalogs.augment_cat_bands(cat, catalog='gaiadr2', verbose=False)

        # Should have added Johnson-Cousins magnitudes
        assert 'Bmag' in result.colnames
        assert 'Vmag' in result.colnames
        assert 'Rmag' in result.colnames
        assert 'Imag' in result.colnames

        # Should have added PS1 conversions
        assert 'gmag' in result.colnames
        assert 'rmag' in result.colnames

    @pytest.mark.unit
    def test_augment_auto_detect(self):
        """Test that auto-detection works when catalog parameter is None."""
        cat = Table()
        cat['gmag'] = np.array([15.0])
        cat['rmag'] = np.array([14.5])
        cat['imag'] = np.array([14.0])
        cat['zmag'] = np.array([13.8])
        cat['ymag'] = np.array([13.7])
        cat['e_gmag'] = np.array([0.01])
        cat['e_rmag'] = np.array([0.01])
        cat['e_imag'] = np.array([0.01])

        # Don't specify catalog - should auto-detect as PS1
        result = catalogs.augment_cat_bands(cat, catalog=None, verbose=False)

        # Should have detected PS1 and augmented accordingly
        assert 'Bmag' in result.colnames
        assert 'Vmag' in result.colnames

    @pytest.mark.unit
    def test_augment_gaiadr2_vizier_column_name(self):
        """Test Gaia DR2 augmentation with the column name Vizier actually returns."""
        cat = Table()
        cat['Gmag'] = np.array([15.0, 16.0])
        cat['BPmag'] = np.array([15.5, 16.5])
        cat['RPmag'] = np.array([14.5, 15.5])
        cat['E(BR/RP)'] = np.array([1.2, 1.3])  # Unsanitized name, as Vizier returns it

        result = catalogs.augment_cat_bands(cat, catalog='gaiadr2', verbose=False)

        for _ in ['Bmag', 'Vmag', 'Rmag', 'Imag']:
            assert _ in result.colnames
            assert np.all(np.isfinite(result[_]))

    @pytest.mark.unit
    def test_detect_gaiaedr3_not_gaiadr2(self):
        """Test that Gaia EDR3 is not mistaken for DR2, which shares its magnitudes."""
        cat = Table()
        cat['Gmag'] = np.array([15.0, 16.0])
        cat['BPmag'] = np.array([15.5, 16.5])
        cat['RPmag'] = np.array([14.5, 15.5])
        cat['RUWE'] = np.array([1.0, 1.1])  # Present in EDR3, absent in DR2

        assert catalogs._detect_catalog_type(cat) == 'gaiaedr3'

    @pytest.mark.unit
    def test_augment_gaiaedr3_no_conversions(self):
        """Test that a known catalog without conversions is left alone quietly."""
        cat = Table()
        cat['Gmag'] = np.array([15.0, 16.0])
        cat['RUWE'] = np.array([1.0, 1.1])

        messages = []
        result = catalogs.augment_cat_bands(cat, catalog='gaiaedr3', verbose=messages.append)

        # No DR2 conversions should have been applied to EDR3 magnitudes
        assert 'Vmag' not in result.colnames
        assert result.colnames == ['Gmag', 'RUWE']
        # And it should not be reported as unknown
        assert not any('Unknown catalog' in _ for _ in messages)

    @pytest.mark.unit
    def test_augment_unknown_catalog(self):
        """Test that unknown catalog type is handled gracefully."""
        cat = Table()
        cat['random_mag'] = [15.0]

        # Unknown catalog should return catalog unchanged
        result = catalogs.augment_cat_bands(cat, catalog='unknown_catalog', verbose=False)

        # Should still return the catalog
        assert 'random_mag' in result.colnames
        # Should not crash

    @pytest.mark.unit
    def test_augment_preserves_original_columns(self):
        """Test that augmentation preserves original columns."""
        cat = Table()
        cat['RAJ2000'] = [180.0]
        cat['DEJ2000'] = [45.0]
        cat['gmag'] = [15.0]
        cat['rmag'] = [14.5]
        cat['imag'] = [14.0]
        cat['zmag'] = [13.8]
        cat['ymag'] = [13.7]

        orig_columns = set(cat.colnames)

        result = catalogs.augment_cat_bands(cat, catalog='ps1', verbose=False)

        # All original columns should still be present
        for col in orig_columns:
            assert col in result.colnames

    @pytest.mark.unit
    def test_augment_atlas_uses_ps1(self):
        """Test that ATLAS catalogs use PS1 augmentation."""
        cat = Table()
        cat['gmag'] = np.array([15.0])
        cat['rmag'] = np.array([14.5])
        cat['imag'] = np.array([14.0])
        cat['zmag'] = np.array([13.8])
        cat['ymag'] = np.array([13.7])
        cat['e_gmag'] = np.array([0.01])
        cat['e_rmag'] = np.array([0.01])
        cat['e_imag'] = np.array([0.01])

        # ATLAS should be normalized to PS1
        result = catalogs.augment_cat_bands(cat, catalog='atlas', verbose=False)

        # Should have PS1-style augmentation
        assert 'Bmag' in result.colnames
        assert 'Vmag' in result.colnames

    @pytest.mark.unit
    def test_augment_missing_columns_handled(self):
        """Test that missing columns are handled gracefully."""
        cat = Table()
        cat['gmag'] = [15.0]
        # Missing other required columns

        # Should not crash - helper functions have try/except
        result = catalogs.augment_cat_bands(cat, catalog='ps1', verbose=False)

        # Should still return catalog
        assert 'gmag' in result.colnames


DEFAULT_SERVER = 'vizier.cds.unistra.fr'
MIRROR = 'vizier.cfa.harvard.edu'


ERROR_VOTABLE = b"""<?xml version="1.0"?>
<VOTABLE><RESOURCE><TABLE name="I/360/syntphot"></TABLE>
<INFO name="Error" value="NOT AVAILABLE YET"/>
<INFO name="Error" value="NOT AVAILABLE YET"/>
<INFO name="QUERY_STATUS" value="ERROR">NOT AVAILABLE YET</INFO>
</RESOURCE></VOTABLE>"""

# Error reply carrying only Error INFOs, without QUERY_STATUS
BARE_ERROR_VOTABLE = b"""<?xml version="1.0"?>
<VOTABLE><RESOURCE><TABLE name="I/360/syntphot"></TABLE>
<INFO name="Error" value="NOT AVAILABLE YET"/>
</RESOURCE></VOTABLE>"""

# Truncated result of a large catalogue as returned by IDIA: rows plus OVERFLOW, and a
# spurious ERROR from the extraction process being interrupted at the row limit
OVERFLOW_VOTABLE = b"""<?xml version="1.0"?>
<VOTABLE><RESOURCE><TABLE name="V/154/sdss16"></TABLE>
<INFO name="Warning" value="truncated result (maxtup=200)"/>
<INFO name="QUERY_STATUS" value="OVERFLOW">truncated result (maxtup=200)</INFO>
<INFO name="Error" value="(pipe) Request has been interrupted"/>
<INFO name="QUERY_STATUS" value="ERROR">(pipe) Request has been interrupted</INFO>
</RESOURCE></VOTABLE>"""


def _fake_vizier(behaviour):
    """Stand-in for astroquery Vizier.

    `behaviour` maps a server name to what its query_region_async does: an exception
    instance to raise, 'empty' to return no tables, 'error' to return a VOTable with
    QUERY_STATUS="ERROR", 'bare_error' to return a VOTable with an Error INFO but no
    QUERY_STATUS, 'html' to return a non-VOTable page, 'overflow' to return one
    table truncated at the row limit (OVERFLOW plus an interrupted-pipe ERROR), or 'ok' to
    return one table.
    Every call is recorded as (server, cache) in the returned class' `calls`, and every
    removal of a cached response as server in `removed`.
    """

    class FakeResponse:
        def __init__(self, server, result):
            self.server = server
            self.result = result
            self.content = {
                'error': ERROR_VOTABLE,
                'bare_error': BARE_ERROR_VOTABLE,
                'overflow': OVERFLOW_VOTABLE,
                'html': b'<html></html>',
            }.get(
                result, b'<?xml version="1.0"?><VOTABLE/>'
            )

        def raise_for_status(self):
            pass

    class FakeQuery:
        def __init__(self, server):
            self.server = server

        def remove_cache_file(self, cache_location):
            FakeVizier.removed.append(self.server)

    class FakeVizier:
        calls = []
        kwargs = []
        removed = []
        cache_location = None

        def __init__(self, *args, **kwargs):
            FakeVizier.kwargs.append(kwargs)
            self.VIZIER_SERVER = DEFAULT_SERVER

        def query_region_async(self, *args, cache=True, **kwargs):
            FakeVizier.calls.append((self.VIZIER_SERVER, cache))
            self._last_query = FakeQuery(self.VIZIER_SERVER)
            result = behaviour[self.VIZIER_SERVER]
            if isinstance(result, Exception):
                raise result
            return FakeResponse(self.VIZIER_SERVER, result)

        def _parse_result(self, response):
            if response.result == 'html':
                return None
            if response.result in ('empty', 'error', 'bare_error'):
                return []
            cat = Table()
            cat['RAJ2000'] = [10.0]
            cat['DEJ2000'] = [20.0]
            return [cat]

    return FakeVizier


class TestVizierMirrors:
    """Test fallback to VizieR mirrors when the default server fails."""

    def _query(self, monkeypatch, behaviour):
        fake = _fake_vizier(behaviour)
        monkeypatch.setattr(catalogs, 'Vizier', fake)
        monkeypatch.setattr(catalogs, 'vizier_mirrors', [MIRROR])
        # These tests cover mirror-selection logic only; the non-VizieR fallback
        # layer is covered separately in TestCatalogFallbacks.
        monkeypatch.setattr(catalogs, 'catalog_fallbacks', {})
        cat = catalogs.get_cat_vizier(10.0, 20.0, 0.1, 'ps1', augment_bands=False)
        return cat, fake.calls

    @pytest.mark.unit
    def test_empty_result_does_not_use_mirror(self, monkeypatch):
        cat, calls = self._query(monkeypatch, {DEFAULT_SERVER: 'empty', MIRROR: 'ok'})

        assert cat is None
        assert calls == [(DEFAULT_SERVER, True),(DEFAULT_SERVER, False)]

    @pytest.mark.unit
    def test_network_error_falls_back_to_mirror(self, monkeypatch):
        error = requests.exceptions.ConnectionError('Connection reset by peer')
        cat, calls = self._query(monkeypatch, {DEFAULT_SERVER: error, MIRROR: 'ok'})

        assert cat is not None
        assert len(cat) == 1
        assert calls == [(DEFAULT_SERVER, True), (MIRROR, True)]

    @pytest.mark.unit
    def test_unparseable_response_falls_back_to_mirror(self, monkeypatch):
        error = TableParseError('Failed to parse VIZIER result!')
        cat, calls = self._query(monkeypatch, {DEFAULT_SERVER: error, MIRROR: 'ok'})

        assert cat is not None
        assert calls == [(DEFAULT_SERVER, True), (MIRROR, True)]

    @pytest.mark.unit
    def test_error_status_falls_back_to_mirror(self, monkeypatch):
        fake = _fake_vizier({DEFAULT_SERVER: 'error', MIRROR: 'ok'})
        monkeypatch.setattr(catalogs, 'Vizier', fake)
        monkeypatch.setattr(catalogs, 'vizier_mirrors', [MIRROR])
        cat = catalogs.get_cat_vizier(10.0, 20.0, 0.1, 'ps1', augment_bands=False)

        assert cat is not None
        assert fake.calls == [(DEFAULT_SERVER, True), (MIRROR, True)]
        # The failed answer must not stay in the cache
        assert fake.removed == [DEFAULT_SERVER]

    @pytest.mark.unit
    def test_error_status_tries_next_mirror(self, monkeypatch):
        second = 'vizier.idia.ac.za'
        error = requests.exceptions.ConnectTimeout('timed out')
        fake = _fake_vizier({DEFAULT_SERVER: error, MIRROR: 'error', second: 'ok'})
        monkeypatch.setattr(catalogs, 'Vizier', fake)
        monkeypatch.setattr(catalogs, 'vizier_mirrors', [MIRROR, second])
        cat = catalogs.get_cat_vizier(10.0, 20.0, 0.1, 'gaiadr3syn', augment_bands=False)

        assert cat is not None
        assert fake.calls == [(DEFAULT_SERVER, True), (MIRROR, True), (second, True)]

    @pytest.mark.unit
    def test_non_votable_response_falls_back_to_mirror(self, monkeypatch):
        cat, calls = self._query(monkeypatch, {DEFAULT_SERVER: 'html', MIRROR: 'ok'})

        assert cat is not None
        assert calls == [(DEFAULT_SERVER, True), (MIRROR, True)]

    @pytest.mark.unit
    def test_vizier_error_messages(self):
        fake = _fake_vizier({DEFAULT_SERVER: 'error'})
        with pytest.raises(catalogs.VizierQueryError, match='^NOT AVAILABLE YET$'):
            catalogs._query_vizier_region(fake(), None, None, 'I/360/syntphot')

    @pytest.mark.unit
    def test_bare_error_info_falls_back_to_mirror(self, monkeypatch):
        second = 'vizier.idia.ac.za'
        error = requests.exceptions.ConnectTimeout('timed out')
        fake = _fake_vizier({DEFAULT_SERVER: error, MIRROR: 'bare_error', second: 'ok'})
        monkeypatch.setattr(catalogs, 'Vizier', fake)
        monkeypatch.setattr(catalogs, 'vizier_mirrors', [MIRROR, second])
        cat = catalogs.get_cat_vizier(10.0, 20.0, 0.1, 'gaiadr3syn', augment_bands=False)

        assert cat is not None
        assert fake.calls == [(DEFAULT_SERVER, True), (MIRROR, True), (second, True)]
        assert fake.removed == [MIRROR]

    @pytest.mark.unit
    def test_truncated_result_is_accepted(self, monkeypatch):
        cat, calls = self._query(monkeypatch, {DEFAULT_SERVER: 'overflow', MIRROR: 'ok'})

        assert cat is not None
        assert len(cat) == 1
        assert calls == [(DEFAULT_SERVER, True)]

    @pytest.mark.unit
    def test_other_errors_propagate(self, monkeypatch):
        with pytest.raises(ValueError):
            self._query(monkeypatch, {DEFAULT_SERVER: ValueError('bad filter'), MIRROR: 'ok'})

    @pytest.mark.unit
    def test_unknown_catalogue_propagates(self, monkeypatch):
        with pytest.raises(EmptyResponseError):
            self._query(monkeypatch, {DEFAULT_SERVER: EmptyResponseError('not found'), MIRROR: 'ok'})

    @pytest.mark.unit
    def test_timeout_passed_to_every_server(self, monkeypatch):
        fake = _fake_vizier({DEFAULT_SERVER: requests.exceptions.Timeout('timed out'), MIRROR: 'ok'})
        monkeypatch.setattr(catalogs, 'Vizier', fake)
        monkeypatch.setattr(catalogs, 'vizier_mirrors', [MIRROR])
        catalogs.get_cat_vizier(10.0, 20.0, 0.1, 'ps1', augment_bands=False, timeout=(3, 30))

        assert [kw['timeout'] for kw in fake.kwargs] == [(3, 30), (3, 30)]

    @pytest.mark.unit
    def test_returns_none_when_all_servers_fail(self, monkeypatch):
        error = requests.exceptions.Timeout('timed out')
        cat, calls = self._query(monkeypatch, {DEFAULT_SERVER: error, MIRROR: error})

        assert cat is None
        assert calls == [(DEFAULT_SERVER, True), (MIRROR, True)]


class TestCoerceNumeric:
    """Test the robust numeric coercion used by the CSV-based fallback sources
    (PS1, SkyMapper) to survive a stray non-numeric value such as an embedded
    upper-limit flag, which VizieR itself parses per-catalogue."""

    @pytest.mark.unit
    def test_clean_column_unaffected(self):
        result = catalogs._coerce_numeric(np.array([1.0, 2.0, 3.0]))

        assert list(result) == [1.0, 2.0, 3.0]
        assert not np.any(result.mask)

    @pytest.mark.unit
    def test_non_numeric_entry_is_masked_not_raised(self):
        result = catalogs._coerce_numeric(np.array(['15.0', '<99.0', '14.0'], dtype=object))

        assert result[0] == 15.0
        assert result[2] == 14.0
        assert bool(result.mask[1])

    @pytest.mark.unit
    def test_none_and_masked_entries_are_masked(self):
        result = catalogs._coerce_numeric([1.0, None, 3.0])

        assert result[0] == 1.0
        assert bool(result.mask[1])
        assert result[2] == 3.0


class TestApplyFilters:
    """Test the client-side filter re-implementation used for fallback catalogues."""

    @pytest.mark.unit
    @pytest.mark.parametrize('expr,expected', [
        ('<15', [10.0, 12.0]),
        ('<=12', [10.0, 12.0]),
        ('>15', [20.0]),
        ('>=15', [15.0, 20.0]),
        ('==15', [15.0]),
        ('!=15', [10.0, 12.0, 20.0]),
    ])
    def test_operators(self, expr, expected):
        cat = Table()
        cat['rmag'] = [10.0, 12.0, 15.0, 20.0]

        result = catalogs._apply_filters(cat, {'rmag': expr})

        assert sorted(result['rmag'].tolist()) == sorted(expected)

    @pytest.mark.unit
    def test_missing_column_is_ignored(self):
        cat = Table()
        cat['rmag'] = [10.0, 20.0]

        result = catalogs._apply_filters(cat, {'gmag': '<15'})

        assert len(result) == 2

    @pytest.mark.unit
    def test_missing_column_logs_a_warning(self):
        cat = Table()
        cat['rmag'] = [10.0, 20.0]
        messages = []

        catalogs._apply_filters(cat, {'gmag': '<15'}, verbose=messages.append)

        assert any('gmag' in str(m) for m in messages)

    @pytest.mark.unit
    def test_non_numeric_column_is_ignored_not_raised(self):
        cat = Table()
        cat['Qflg'] = ['AAA', 'BBB']
        messages = []

        result = catalogs._apply_filters(cat, {'Qflg': '<15'}, verbose=messages.append)

        assert len(result) == 2
        assert any('Qflg' in str(m) for m in messages)

    @pytest.mark.unit
    def test_unparseable_expression_is_ignored(self):
        cat = Table()
        cat['rmag'] = [10.0, 20.0]

        result = catalogs._apply_filters(cat, {'rmag': 'not a filter'})

        assert len(result) == 2

    @pytest.mark.unit
    def test_no_filters(self):
        cat = Table()
        cat['rmag'] = [10.0, 20.0]

        assert len(catalogs._apply_filters(cat, {})) == 2
        assert len(catalogs._apply_filters(cat, None)) == 2


_FAKE_VOTABLE = '''<?xml version="1.0"?>
<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
<RESOURCE type="results">
<TABLE>
<FIELD name="ra" datatype="double" ID="col_0"/>
<FIELD name="dec" datatype="double" ID="col_1"/>
<DATA><TABLEDATA>
<TR><TD>10.0</TD><TD>20.0</TD></TR>
</TABLEDATA></DATA>
</TABLE>
</RESOURCE>
</VOTABLE>'''


class _FakeTapResponse:
    def __init__(self, status_code=200, headers=None, text='', content=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self.content = content if content is not None else text.encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f'{self.status_code} error')


class _FakeTapSession:
    """Scripts the IVOA UWS exchange (submit -> poll -> fetch) that
    _tap_async_query drives, without any network access."""

    def __init__(self, phases, error_text='', votable=_FAKE_VOTABLE):
        self.phases = list(phases)
        self.error_text = error_text
        self.votable = votable
        self.aborted = False
        self.post_calls = []
        self.get_calls = []

    def post(self, url, data=None, timeout=None, allow_redirects=True):
        self.post_calls.append((url, data, timeout))
        if url.endswith('/async'):
            return _FakeTapResponse(status_code=303, headers={'Location': 'http://example.test/async/job1'})
        if (data or {}).get('PHASE') == 'ABORT':
            self.aborted = True
            return _FakeTapResponse(status_code=200, text='ABORTED')
        raise AssertionError(f'unexpected POST {url}')

    def get(self, url, timeout=None):
        self.get_calls.append((url, timeout))
        if url.endswith('/phase'):
            phase = self.phases.pop(0) if self.phases else 'EXECUTING'
            return _FakeTapResponse(status_code=200, text=phase)
        if url.endswith('/error'):
            return _FakeTapResponse(status_code=200, text=self.error_text)
        if url.endswith('/results/result'):
            return _FakeTapResponse(status_code=200, content=self.votable.encode())
        raise AssertionError(f'unexpected GET {url}')

    def close(self):
        pass


class TestTapAsyncQuery:
    """Protocol-level tests for the hand-rolled IVOA UWS async client that
    replaced GaiaClass/pyvo's async job wait, neither of which track an
    overall deadline across polls (verified by reading their source)."""

    @pytest.mark.unit
    def test_happy_path_returns_parsed_table(self, monkeypatch):
        session = _FakeTapSession(phases=['COMPLETED'])
        monkeypatch.setattr(catalogs.requests, 'Session', lambda: session)

        result = catalogs._tap_async_query('http://example.test', 'SELECT 1', timeout=(5, 10))

        assert result.colnames == ['ra', 'dec']
        assert result['ra'][0] == 10.0

    @pytest.mark.unit
    def test_error_phase_raises_and_logs(self, monkeypatch):
        session = _FakeTapSession(phases=['ERROR'], error_text='bad query: syntax error')
        monkeypatch.setattr(catalogs.requests, 'Session', lambda: session)
        messages = []

        with pytest.raises(RuntimeError, match='ERROR'):
            catalogs._tap_async_query('http://example.test', 'SELECT bad', timeout=(5, 10), verbose=messages.append)

        assert any('bad query' in str(m) for m in messages)

    @pytest.mark.unit
    def test_timeout_raises_and_aborts_with_warning(self, monkeypatch):
        # the job never reaches a terminal phase, so this must give up once
        # the deadline passes rather than polling forever
        session = _FakeTapSession(phases=[])
        monkeypatch.setattr(catalogs.requests, 'Session', lambda: session)
        messages = []

        with pytest.raises(TimeoutError):
            catalogs._tap_async_query('http://example.test', 'SELECT 1', timeout=(0.1, 0.3), verbose=messages.append)

        assert session.aborted
        assert any('timeout' in str(m).lower() or 'did not complete' in str(m).lower() for m in messages)
        assert any('abort' in str(m).lower() for m in messages)


def _one_row_cat(ra, dec):
    """A minimal one-row Table, used as a stand-in fallback result to tell apart
    'fallback data' from 'real VizieR data' in interaction tests below."""
    cat = Table()
    cat['RAJ2000'] = [ra]
    cat['DEJ2000'] = [dec]
    return cat


class TestFallbackDispatch:
    """Test that get_cat_vizier reaches for a registered fallback only when VizieR
    genuinely failed to answer - never for a confirmed-empty result (all mocked)."""

    def _query(self, monkeypatch, fallback_fn, catalog='ps1', **kwargs):
        error = requests.exceptions.ConnectionError('Connection reset by peer')
        fake = _fake_vizier({DEFAULT_SERVER: error, MIRROR: error})
        monkeypatch.setattr(catalogs, 'Vizier', fake)
        monkeypatch.setattr(catalogs, 'vizier_mirrors', [MIRROR])
        monkeypatch.setattr(catalogs, 'catalog_fallbacks', {catalog: fallback_fn})
        return catalogs.get_cat_vizier(10.0, 20.0, 0.1, catalog, augment_bands=False, **kwargs)

    @pytest.mark.unit
    def test_fallback_used_when_mirrors_exhausted(self, monkeypatch):
        def fake_fallback(ra0, dec0, sr0, filters=None, timeout=None, verbose=False):
            cat = Table()
            cat['RAJ2000'] = [10.0, 10.1]
            cat['DEJ2000'] = [20.0, 20.1]
            cat['rmag'] = [14.0, 19.0]
            return cat

        cat = self._query(monkeypatch, fake_fallback)

        assert cat is not None
        assert len(cat) == 2
        assert cat.meta['vizier_id'] == catalogs.catalogs['ps1']['vizier']

    @pytest.mark.unit
    def test_fallback_result_is_filtered_and_limited(self, monkeypatch):
        def fake_fallback(ra0, dec0, sr0, filters=None, timeout=None, verbose=False):
            cat = Table()
            cat['RAJ2000'] = [10.0, 10.1, 10.2]
            cat['DEJ2000'] = [20.0, 20.1, 20.2]
            cat['rmag'] = [14.0, 16.0, 19.0]
            return cat

        cat = self._query(monkeypatch, fake_fallback, filters={'rmag': '<18'}, limit=1)

        assert len(cat) == 1
        assert cat['rmag'][0] < 18

    @pytest.mark.unit
    def test_returns_none_when_fallback_returns_none(self, monkeypatch):
        cat = self._query(monkeypatch, lambda *a, **kw: None)

        assert cat is None

    @pytest.mark.unit
    def test_returns_none_when_fallback_raises(self, monkeypatch):
        def broken_fallback(*a, **kw):
            raise RuntimeError('schema changed upstream')

        cat = self._query(monkeypatch, broken_fallback)

        assert cat is None

    @pytest.mark.unit
    def test_no_fallback_registered_returns_none(self, monkeypatch):
        error = requests.exceptions.ConnectionError('Connection reset by peer')
        fake = _fake_vizier({DEFAULT_SERVER: error, MIRROR: error})
        monkeypatch.setattr(catalogs, 'Vizier', fake)
        monkeypatch.setattr(catalogs, 'vizier_mirrors', [MIRROR])
        monkeypatch.setattr(catalogs, 'catalog_fallbacks', {})

        cat = catalogs.get_cat_vizier(10.0, 20.0, 0.1, 'atlas', augment_bands=False)

        assert cat is None

    @pytest.mark.unit
    def test_fallback_reached_only_after_every_mirror_genuinely_errors(self, monkeypatch):
        """All three servers report QUERY_STATUS=ERROR (not a clean empty result):
        the mirror loop must actually try all of them - not stop at the first
        false-empty response - before the fallback is used."""
        second = 'vizier.idia.ac.za'
        fake = _fake_vizier({DEFAULT_SERVER: 'error', MIRROR: 'error', second: 'error'})
        monkeypatch.setattr(catalogs, 'Vizier', fake)
        monkeypatch.setattr(catalogs, 'vizier_mirrors', [MIRROR, second])
        monkeypatch.setattr(catalogs, 'catalog_fallbacks', {'ps1': lambda *a, **kw: _one_row_cat(1.0, 2.0)})

        cat = catalogs.get_cat_vizier(10.0, 20.0, 0.1, 'ps1', augment_bands=False)

        assert cat is not None and cat['RAJ2000'][0] == 1.0
        assert fake.calls == [(DEFAULT_SERVER, True), (MIRROR, True), (second, True)]

    @pytest.mark.unit
    def test_fallback_not_used_when_a_later_mirror_has_real_data(self, monkeypatch):
        """The first server errors but the mirror has real data: the fallback
        must not override genuine VizieR data found on a later mirror."""
        fake = _fake_vizier({DEFAULT_SERVER: 'error', MIRROR: 'ok'})
        monkeypatch.setattr(catalogs, 'Vizier', fake)
        monkeypatch.setattr(catalogs, 'vizier_mirrors', [MIRROR])
        monkeypatch.setattr(catalogs, 'catalog_fallbacks', {'ps1': lambda *a, **kw: _one_row_cat(1.0, 2.0)})

        cat = catalogs.get_cat_vizier(10.0, 20.0, 0.1, 'ps1', augment_bands=False)

        assert cat is not None and cat['RAJ2000'][0] == 10.0  # real VizieR data, not the 1.0 fallback stub

    @pytest.mark.unit
    def test_confirmed_empty_result_skips_fallback(self, monkeypatch):
        """A clean empty result (no error, server genuinely answered) stops the
        mirror loop by design (see TestVizierMirrors.test_empty_result_does_not_use_mirror)
        - and must NOT fall through to a non-VizieR source either: VizieR confirming
        nothing is there is trustworthy, unlike VizieR failing to answer at all.
        Falling back here would risk padding a genuinely dark patch of sky with
        whatever an unrelated survey happens to cover at that position."""
        fake = _fake_vizier({DEFAULT_SERVER: 'empty', MIRROR: 'ok'})
        monkeypatch.setattr(catalogs, 'Vizier', fake)
        monkeypatch.setattr(catalogs, 'vizier_mirrors', [MIRROR])
        monkeypatch.setattr(catalogs, 'catalog_fallbacks', {'ps1': lambda *a, **kw: _one_row_cat(1.0, 2.0)})

        cat = catalogs.get_cat_vizier(10.0, 20.0, 0.1, 'ps1', augment_bands=False)

        assert cat is None
        assert fake.calls == [(DEFAULT_SERVER, True),(DEFAULT_SERVER, False)]


class _FakeResponse:
    """Stand-in for requests.Response, used by the PS1 and SkyMapper fallbacks."""

    def __init__(self, text):
        self.text = text
        self.content = text.encode()

    def raise_for_status(self):
        pass


class TestFallbackSources:
    """Column-mapping regression tests for each non-VizieR fallback, with every
    external call mocked so no network access is required."""

    @pytest.mark.unit
    def test_fallback_ps1_maps_columns_and_masks_missing(self, monkeypatch):
        csv = (
            'raMean,decMean,raMeanErr,decMeanErr,'
            'gMeanPSFMag,gMeanPSFMagErr,rMeanPSFMag,rMeanPSFMagErr,'
            'iMeanPSFMag,iMeanPSFMagErr,zMeanPSFMag,zMeanPSFMagErr,'
            'yMeanPSFMag,yMeanPSFMagErr\n'
            '10.0,20.0,0.001,0.001,15.0,0.01,14.5,0.01,-999,-999,13.8,0.01,13.7,0.01\n'
        )
        monkeypatch.setattr(catalogs.requests, 'get', lambda *a, **kw: _FakeResponse(csv))

        cat = catalogs._fallback_ps1(10.0, 20.0, 0.1)

        assert cat is not None
        assert cat['RAJ2000'][0] == 10.0 and cat['DEJ2000'][0] == 20.0
        assert cat['gmag'][0] == 15.0
        assert bool(cat['imag'].mask[0])  # -999 sentinel masked out

    @pytest.mark.unit
    def test_fallback_ps1_requires_at_least_three_detections(self, monkeypatch):
        captured = {}

        def fake_get(url, params=None, timeout=None):
            captured.update(params)
            csv = (
                'raMean,decMean,raMeanErr,decMeanErr,'
                'gMeanPSFMag,gMeanPSFMagErr,rMeanPSFMag,rMeanPSFMagErr,'
                'iMeanPSFMag,iMeanPSFMagErr,zMeanPSFMag,zMeanPSFMagErr,'
                'yMeanPSFMag,yMeanPSFMagErr\n'
                '10.0,20.0,0.001,0.001,15.0,0.01,14.5,0.01,14.0,0.01,13.8,0.01,13.7,0.01\n'
            )
            return _FakeResponse(csv)

        monkeypatch.setattr(catalogs.requests, 'get', fake_get)

        catalogs._fallback_ps1(10.0, 20.0, 0.1)

        assert captured.get('nDetections.gte') == 3

    @pytest.mark.unit
    def test_fallback_ps1_pushes_filter_to_mast_params(self, monkeypatch):
        """A simple magnitude-limit filter must be forwarded to MAST as a native
        '<column>.<decorator>' query param, not just applied after the fixed-size
        page is fetched - otherwise coverage silently collapses in dense fields
        once the unfiltered page fills up before reaching the cut."""
        captured = {}

        def fake_get(url, params=None, timeout=None):
            captured.update(params)
            csv = (
                'raMean,decMean,raMeanErr,decMeanErr,'
                'gMeanPSFMag,gMeanPSFMagErr,rMeanPSFMag,rMeanPSFMagErr,'
                'iMeanPSFMag,iMeanPSFMagErr,zMeanPSFMag,zMeanPSFMagErr,'
                'yMeanPSFMag,yMeanPSFMagErr\n'
                '10.0,20.0,0.001,0.001,15.0,0.01,14.5,0.01,14.0,0.01,13.8,0.01,13.7,0.01\n'
            )
            return _FakeResponse(csv)

        monkeypatch.setattr(catalogs.requests, 'get', fake_get)

        cat = catalogs._fallback_ps1(10.0, 20.0, 0.1, filters={'rmag': '<20.0'})

        assert cat is not None
        assert captured.get('rMeanPSFMag.lt') == '20.0'
        assert captured.get('rMeanPSFMag.gt') == '0'  # excludes MAST's -999 sentinel

    @pytest.mark.unit
    def test_fallback_ps1_ignores_filter_on_unmapped_column(self, monkeypatch):
        captured = {}

        def fake_get(url, params=None, timeout=None):
            captured.update(params)
            csv = (
                'raMean,decMean,raMeanErr,decMeanErr,'
                'gMeanPSFMag,gMeanPSFMagErr,rMeanPSFMag,rMeanPSFMagErr,'
                'iMeanPSFMag,iMeanPSFMagErr,zMeanPSFMag,zMeanPSFMagErr,'
                'yMeanPSFMag,yMeanPSFMagErr\n'
                '10.0,20.0,0.001,0.001,15.0,0.01,14.5,0.01,14.0,0.01,13.8,0.01,13.7,0.01\n'
            )
            return _FakeResponse(csv)

        monkeypatch.setattr(catalogs.requests, 'get', fake_get)

        cat = catalogs._fallback_ps1(10.0, 20.0, 0.1, filters={'Qual': '<5'})

        assert cat is not None
        assert not any(k.startswith('Qual') for k in captured)

    @pytest.mark.unit
    def test_fallback_ps1_masks_embedded_non_numeric_placeholder(self, monkeypatch):
        """A stray non-numeric value (e.g. an upper-limit flag like '<24.0',
        which VizieR itself would parse per-catalogue) must be handled."""
        csv = (
            'raMean,decMean,raMeanErr,decMeanErr,'
            'gMeanPSFMag,gMeanPSFMagErr,rMeanPSFMag,rMeanPSFMagErr,'
            'iMeanPSFMag,iMeanPSFMagErr,zMeanPSFMag,zMeanPSFMagErr,'
            'yMeanPSFMag,yMeanPSFMagErr\n'
            '10.0,20.0,0.001,0.001,15.0,0.01,14.5,0.01,14.0,0.01,13.8,0.01,13.7,0.01\n'
            '10.1,20.1,0.001,0.001,<24.0,0.01,14.6,0.01,14.1,0.01,13.9,0.01,13.6,0.01\n'
        )
        monkeypatch.setattr(catalogs.requests, 'get', lambda *a, **kw: _FakeResponse(csv))

        cat = catalogs._fallback_ps1(10.0, 20.0, 0.1)

        assert cat is not None
        assert len(cat) == 2
        assert cat['gmag'][0] == 15.0
        assert bool(cat['gmag'].mask[1])

    @staticmethod
    def _gaia_source_row(ra_j2000=10.0, dec_j2000=20.0):
        """A minimal gaiadr3.gaia_source-shaped row for the eDR3 fallback tests
        (ra_j2000/dec_j2000 stand in for the server-side EPOCH_PROP_POS result)."""
        raw = Table()
        raw['source_id'] = [1]
        raw['ra_j2000'] = [ra_j2000]
        raw['dec_j2000'] = [dec_j2000]
        raw['ra_error'] = [0.1]
        raw['dec_error'] = [0.1]
        raw['parallax'] = [5.0]
        raw['parallax_error'] = [0.1]
        raw['pmra'] = [12.3]
        raw['pmra_error'] = [0.05]
        raw['pmdec'] = [-4.5]
        raw['pmdec_error'] = [0.05]
        raw['phot_g_mean_mag'] = [15.0]
        raw['phot_g_mean_flux'] = [1.0e5]
        raw['phot_g_mean_flux_error'] = [100.0]
        raw['phot_bp_mean_mag'] = [15.5]
        raw['phot_bp_mean_flux'] = [6.0e4]
        raw['phot_bp_mean_flux_error'] = [80.0]
        raw['phot_rp_mean_mag'] = [14.5]
        raw['phot_rp_mean_flux'] = [1.4e5]
        raw['phot_rp_mean_flux_error'] = [120.0]
        raw['ruwe'] = [1.0]
        return raw

    @pytest.mark.unit
    def test_fallback_gaiaedr3_maps_columns(self, monkeypatch):
        raw = self._gaia_source_row()
        monkeypatch.setattr(catalogs, '_tap_async_query', lambda *a, **kw: raw)

        cat = catalogs._fallback_gaiaedr3(10.0, 20.0, 0.1)

        assert cat['Gmag'][0] == 15.0
        assert cat['BPmag'][0] == 15.5
        assert cat['RPmag'][0] == 14.5
        assert cat['RUWE'][0] == 1.0
        assert cat['Plx'][0] == 5.0
        assert cat['pmRA'][0] == 12.3
        assert cat['pmDE'][0] == -4.5
        assert cat['e_Gmag'][0] == pytest.approx(2.5 / np.log(10) * 100.0 / 1.0e5)

    @pytest.mark.unit
    def test_fallback_gaiaedr3_requests_epoch_propagation_in_adql(self, monkeypatch):
        """gaia_source reports ra/dec at ref_epoch (2016.0 for DR3); VizieR's
        RAJ2000/DEJ2000 are propagated back to J2000 using the proper motion.
        The archive's own EPOCH_PROP_POS function does this server-side -
        validated against real VizieR data to ~1e-7" (see source docstring)."""
        raw = self._gaia_source_row(ra_j2000=10.0001, dec_j2000=19.9998)
        captured = {}

        def fake_tap(base_url, query, timeout=(10, 60), verbose=False):
            captured['query'] = query
            return raw

        monkeypatch.setattr(catalogs, '_tap_async_query', fake_tap)

        cat = catalogs._fallback_gaiaedr3(10.0, 20.0, 0.1)

        assert 'EPOCH_PROP_POS(' in captured['query']
        assert 'COORD1(' in captured['query'] and 'COORD2(' in captured['query']
        assert 'ref_epoch' in captured['query']
        # the fallback must report whatever the server propagated, not the raw position
        assert cat['RAJ2000'][0] == 10.0001
        assert cat['DEJ2000'][0] == 19.9998

    @pytest.mark.unit
    def test_fallback_gaiaedr3_pushes_filter_to_adql(self, monkeypatch):
        raw = self._gaia_source_row()
        captured = {}

        def fake_tap(base_url, query, timeout=(10, 60), verbose=False):
            captured['query'] = query
            return raw

        monkeypatch.setattr(catalogs, '_tap_async_query', fake_tap)

        cat = catalogs._fallback_gaiaedr3(10.0, 20.0, 0.1, filters={'Gmag': '<19.0'})

        assert cat is not None
        assert 'phot_g_mean_mag < 19.0' in captured['query']

    @pytest.mark.unit
    def test_fallback_gaiadr3syn_maps_columns(self, monkeypatch):
        raw = Table()
        raw['source_id'] = [1]
        raw['ra'] = [10.0]
        raw['dec'] = [20.0]
        for col in ['u_jkc', 'b_jkc', 'v_jkc', 'r_jkc', 'i_jkc',
                    'u_sdss', 'g_sdss', 'r_sdss', 'i_sdss', 'z_sdss', 'y_ps1']:
            raw[col + '_mag'] = [15.0]
            raw[col + '_flux'] = [100.0]
            raw[col + '_flux_error'] = [1.0]

        monkeypatch.setattr(catalogs, '_tap_async_query', lambda *a, **kw: raw)

        cat = catalogs._fallback_gaiadr3syn(10.0, 20.0, 0.1)

        assert cat['Umag'][0] == 15.0
        assert cat['FU'][0] == 100.0
        assert cat['e_FU'][0] == 1.0
        assert cat['gmag'][0] == 15.0
        assert cat['Fg'][0] == 100.0
        assert cat['Fy'][0] == 100.0
        # Unlike gaiaedr3, VizieR does not propagate I/360/syntphot positions
        # to J2000, so ra/dec are used as-is here.
        assert cat['RAJ2000'][0] == 10.0
        assert cat['DEJ2000'][0] == 20.0

    @pytest.mark.unit
    def test_fallback_gaiadr3syn_pushes_filter_to_adql(self, monkeypatch):
        raw = Table()
        raw['source_id'] = [1]
        raw['ra'] = [10.0]
        raw['dec'] = [20.0]
        for col in ['u_jkc', 'b_jkc', 'v_jkc', 'r_jkc', 'i_jkc',
                    'u_sdss', 'g_sdss', 'r_sdss', 'i_sdss', 'z_sdss', 'y_ps1']:
            raw[col + '_mag'] = [15.0]
            raw[col + '_flux'] = [100.0]
            raw[col + '_flux_error'] = [1.0]
        captured = {}

        def fake_tap(base_url, query, timeout=(10, 60), verbose=False):
            captured['query'] = query
            return raw

        monkeypatch.setattr(catalogs, '_tap_async_query', fake_tap)

        cat = catalogs._fallback_gaiadr3syn(10.0, 20.0, 0.1, filters={'rmag': '<20.0'})

        assert cat is not None
        assert 's.r_sdss_mag < 20.0' in captured['query']

    @pytest.mark.unit
    def test_fallback_skymapper_maps_columns(self, monkeypatch):
        csv = (
            'raj2000,dej2000,e_raj2000,e_dej2000,flags,class_star,'
            'u_psf,e_u_psf,v_psf,e_v_psf,'
            'g_psf,e_g_psf,r_psf,e_r_psf,i_psf,e_i_psf,z_psf,e_z_psf\n'
            '10.0,-20.0,0.001,0.001,3,0.95,16.5,0.02,16.0,0.02,15.0,0.01,14.5,0.01,14.0,0.01,13.8,0.01\n'
        )
        monkeypatch.setattr(catalogs.requests, 'get', lambda *a, **kw: _FakeResponse(csv))

        cat = catalogs._fallback_skymapper(10.0, -20.0, 0.1)

        assert cat['RAJ2000'][0] == 10.0 and cat['DEJ2000'][0] == -20.0
        assert cat['uPSF'][0] == 16.5
        assert cat['e_vPSF'][0] == 0.02
        assert cat['gPSF'][0] == 15.0
        assert cat['e_rPSF'][0] == 0.01
        assert cat['flags'][0] == 3
        assert cat['ClassStar'][0] == 0.95

    @pytest.mark.unit
    def test_fallback_skymapper_pushes_filter_to_adql(self, monkeypatch):
        csv = (
            'raj2000,dej2000,e_raj2000,e_dej2000,flags,class_star,'
            'u_psf,e_u_psf,v_psf,e_v_psf,'
            'g_psf,e_g_psf,r_psf,e_r_psf,i_psf,e_i_psf,z_psf,e_z_psf\n'
            '10.0,-20.0,0.001,0.001,3,0.95,16.5,0.02,16.0,0.02,15.0,0.01,14.5,0.01,14.0,0.01,13.8,0.01\n'
        )
        captured = {}

        def fake_get(url, params=None, timeout=None):
            captured['url'] = url
            captured['query'] = params['QUERY']
            return _FakeResponse(csv)

        monkeypatch.setattr(catalogs.requests, 'get', fake_get)

        cat = catalogs._fallback_skymapper(10.0, -20.0, 0.1, filters={'rPSF': '<18.0'})

        assert cat is not None
        assert captured['url'] == 'https://api.skymapper.nci.org.au/public/tap/sync'
        assert 'r_psf < 18.0' in captured['query']

    @pytest.mark.unit
    def test_fallback_skymapper_warns_when_row_cap_hit(self, monkeypatch):
        header = (
            'raj2000,dej2000,e_raj2000,e_dej2000,flags,class_star,'
            'u_psf,e_u_psf,v_psf,e_v_psf,'
            'g_psf,e_g_psf,r_psf,e_r_psf,i_psf,e_i_psf,z_psf,e_z_psf\n'
        )
        row = '10.0,-20.0,0.001,0.001,3,0.95,16.5,0.02,16.0,0.02,15.0,0.01,14.5,0.01,14.0,0.01,13.8,0.01\n'
        monkeypatch.setattr(catalogs, '_SKYMAPPER_ROW_LIMIT', 2)
        csv = header + row * 2
        monkeypatch.setattr(catalogs.requests, 'get', lambda *a, **kw: _FakeResponse(csv))
        messages = []

        cat = catalogs._fallback_skymapper(10.0, -20.0, 0.1, verbose=messages.append)

        assert len(cat) == 2
        assert any('limit' in str(m).lower() for m in messages)

    @pytest.mark.unit
    def test_fallback_2mass_maps_columns(self, monkeypatch):
        raw = Table()
        raw['ra'] = [10.0]
        raw['dec'] = [20.0]
        raw['designation'] = ['00400119+2000440']
        raw['j_m'] = [14.0]
        raw['j_msigcom'] = [0.02]
        raw['h_m'] = [13.5]
        raw['h_msigcom'] = [0.02]
        raw['k_m'] = [13.0]
        raw['k_msigcom'] = [0.02]
        raw['ph_qual'] = ['AAA']
        raw['rd_flg'] = ['222']
        raw['bl_flg'] = ['111']
        raw['cc_flg'] = ['000']
        raw['gal_contam'] = [0]
        raw['mp_flg'] = [0]

        monkeypatch.setattr(catalogs, '_tap_async_query', lambda *a, **kw: raw)

        cat = catalogs._fallback_2mass(10.0, 20.0, 0.1)

        assert cat['Jmag'][0] == 14.0
        assert cat['Hmag'][0] == 13.5
        assert cat['Kmag'][0] == 13.0
        assert cat['2MASS'][0] == '00400119+2000440'
        assert cat['Qflg'][0] == 'AAA'
        assert cat['Rflg'][0] == '222'
        assert cat['Bflg'][0] == '111'
        assert cat['Cflg'][0] == '000'
        assert cat['Xflg'][0] == 0
        assert cat['Aflg'][0] == 0

    @pytest.mark.unit
    def test_fallback_vhs_maps_columns_and_masks_sentinel(self, monkeypatch):
        import astroquery.vsa as vsa_module
        raw = Table()
        raw['ra'] = [10.0, 10.1]
        raw['dec'] = [-20.0, -20.1]
        raw['jAperMag3'] = [17.0, -999999500.0]
        raw['jAperMag3Err'] = [0.02, -999999500.0]
        raw['hAperMag3'] = [16.5, 16.6]
        raw['hAperMag3Err'] = [0.02, 0.02]
        raw['ksAperMag3'] = [16.0, 16.1]
        raw['ksAperMag3Err'] = [0.02, 0.02]
        raw['mergedClassStat'] = [-120.0, 85.0]
        raw['mergedClass'] = [-1, 1]
        raw['pStar'] = [0.99, 0.05]
        raw['pGalaxy'] = [0.01, 0.9]
        raw['pNoise'] = [0.0, 0.0]
        raw['yErrBits'] = [0, 0]
        raw['yppErrBits'] = [0, 0]
        raw['jErrBits'] = [0, 0]
        raw['jppErrBits'] = [0, 0]
        raw['hErrBits'] = [0, 0]
        raw['hppErrBits'] = [0, 0]
        raw['ksErrBits'] = [0, 0]
        raw['ksppErrBits'] = [0, 0]

        monkeypatch.setattr(vsa_module.Vsa, 'query_region', lambda *a, **kw: raw)

        cat = catalogs._fallback_vhs(10.0, -20.0, 0.1)

        assert cat['Jap3'][0] == 17.0
        assert bool(cat['Jap3'].mask[1])  # VSA sentinel masked out
        assert cat['Ksap3'][0] == 16.0
        # VizieR's exact short names for these (confirmed from its own query form)
        assert cat['Mclass'][0] == -1
        assert cat['Mstat'][0] == -120.0
        assert cat['pStar'][0] == 0.99
        assert cat['pGal'][0] == 0.01
        assert cat['Jerrbits'][0] == 0
        assert cat['Jperrbits'][0] == 0
        assert 'pSaturated' not in cat.colnames  # not exposed by VizieR II/367
        # Must not claim the VizieR-registered DR5 name for DR4 fallback data.
        assert cat.meta['name'] == 'VISTA Hemisphere Survey DR4'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
