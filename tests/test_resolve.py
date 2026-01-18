"""
Tests for resolve.py - object name and coordinate resolution.

These tests avoid network access by mocking external service calls
(Simbad, TNS, Astropy SkyCoord.from_name).

Tests cover all coordinate formats including comma-separated coordinates.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from astropy.coordinates import SkyCoord

from stdpipe import resolve


class TestParseSexadecimal:
    """Test sexagesimal coordinate parsing."""

    def test_parse_space_separated_positive(self):
        """Test parsing space-separated sexagesimal format (positive)."""
        result = resolve.parseSexadecimal("12 34 56.78")
        expected = 12 + 34/60 + 56.78/3600
        assert np.isclose(result, expected, rtol=1e-9)

    def test_parse_space_separated_negative(self):
        """Test parsing space-separated sexagesimal format (negative)."""
        result = resolve.parseSexadecimal("-12 34 56.78")
        expected = -(12 + 34/60 + 56.78/3600)
        assert np.isclose(result, expected, rtol=1e-9)

    def test_parse_colon_separated_positive(self):
        """Test parsing colon-separated sexagesimal format (positive)."""
        result = resolve.parseSexadecimal("12:34:56.78")
        expected = 12 + 34/60 + 56.78/3600
        assert np.isclose(result, expected, rtol=1e-9)

    def test_parse_colon_separated_negative(self):
        """Test parsing colon-separated sexagesimal format (negative)."""
        result = resolve.parseSexadecimal("-12:34:56.78")
        expected = -(12 + 34/60 + 56.78/3600)
        assert np.isclose(result, expected, rtol=1e-9)

    def test_parse_with_plus_sign(self):
        """Test parsing with explicit plus sign."""
        result = resolve.parseSexadecimal("+12 34 56.78")
        expected = 12 + 34/60 + 56.78/3600
        assert np.isclose(result, expected, rtol=1e-9)

    def test_parse_integer_seconds(self):
        """Test parsing with integer seconds (no decimal)."""
        result = resolve.parseSexadecimal("12 34 56")
        expected = 12 + 34/60 + 56/3600
        assert np.isclose(result, expected, rtol=1e-9)

    def test_parse_with_whitespace(self):
        """Test parsing with extra whitespace."""
        result = resolve.parseSexadecimal("  12  34  56.78  ")
        expected = 12 + 34/60 + 56.78/3600
        assert np.isclose(result, expected, rtol=1e-9)

    def test_parse_zero_degrees(self):
        """Test parsing zero degrees."""
        result = resolve.parseSexadecimal("0 0 0")
        assert result == 0.0

    def test_parse_zero_negative(self):
        """Test parsing negative zero (edge case)."""
        result = resolve.parseSexadecimal("-0 0 0")
        assert result == 0.0  # -0 becomes 0

    def test_parse_max_declination(self):
        """Test parsing maximum declination (90 degrees)."""
        result = resolve.parseSexadecimal("90 0 0")
        assert result == 90.0

    def test_parse_invalid_format_returns_zero(self):
        """Test that invalid format returns 0."""
        result = resolve.parseSexadecimal("invalid string")
        assert result == 0

    def test_parse_partial_format_returns_zero(self):
        """Test that partial format returns 0."""
        result = resolve.parseSexadecimal("12 34")  # Missing seconds
        assert result == 0

    def test_parse_three_digit_degrees(self):
        """Test parsing with three-digit degrees (for RA)."""
        result = resolve.parseSexadecimal("123 45 67.89")
        expected = 123 + 45/60 + 67.89/3600
        assert np.isclose(result, expected, rtol=1e-9)


class TestResolveDecimalDegrees:
    """Test resolve() with decimal degree coordinates (no network)."""

    def test_resolve_decimal_degrees_positive(self):
        """Test resolving two decimal degree values (positive)."""
        target = resolve.resolve("123.456 45.678")
        assert target is not None
        assert np.isclose(target.ra.deg, 123.456, rtol=1e-9)
        assert np.isclose(target.dec.deg, 45.678, rtol=1e-9)

    def test_resolve_decimal_degrees_negative_dec(self):
        """Test resolving with negative declination."""
        target = resolve.resolve("123.456 -45.678")
        assert target is not None
        assert np.isclose(target.ra.deg, 123.456, rtol=1e-9)
        assert np.isclose(target.dec.deg, -45.678, rtol=1e-9)

    def test_resolve_decimal_degrees_with_whitespace(self):
        """Test resolving with extra whitespace."""
        target = resolve.resolve("  123.456   45.678  ")
        assert target is not None
        assert np.isclose(target.ra.deg, 123.456, rtol=1e-9)
        assert np.isclose(target.dec.deg, 45.678, rtol=1e-9)

    def test_resolve_decimal_degrees_integer(self):
        """Test resolving integer degree values."""
        target = resolve.resolve("123 45")
        assert target is not None
        assert np.isclose(target.ra.deg, 123.0, rtol=1e-9)
        assert np.isclose(target.dec.deg, 45.0, rtol=1e-9)

    def test_resolve_decimal_degrees_zero(self):
        """Test resolving zero coordinates."""
        target = resolve.resolve("0 0")
        assert target is not None
        assert np.isclose(target.ra.deg, 0.0, rtol=1e-9)
        assert np.isclose(target.dec.deg, 0.0, rtol=1e-9)

    def test_resolve_decimal_degrees_360(self):
        """Test resolving RA near 360 degrees."""
        target = resolve.resolve("359.999 89.999")
        assert target is not None
        assert np.isclose(target.ra.deg, 359.999, rtol=1e-9)
        assert np.isclose(target.dec.deg, 89.999, rtol=1e-9)

    def test_resolve_decimal_degrees_comma_separated(self):
        """Test resolving comma-separated decimal degrees."""
        target = resolve.resolve("123.456, 45.678")
        assert target is not None
        assert np.isclose(target.ra.deg, 123.456, rtol=1e-9)
        assert np.isclose(target.dec.deg, 45.678, rtol=1e-9)

    def test_resolve_decimal_degrees_comma_negative_dec(self):
        """Test resolving comma-separated with negative declination."""
        target = resolve.resolve("123.456, -45.678")
        assert target is not None
        assert np.isclose(target.ra.deg, 123.456, rtol=1e-9)
        assert np.isclose(target.dec.deg, -45.678, rtol=1e-9)

    def test_resolve_decimal_degrees_comma_with_spaces(self):
        """Test resolving comma-separated with extra spaces."""
        target = resolve.resolve("123.456 , 45.678")
        assert target is not None
        assert np.isclose(target.ra.deg, 123.456, rtol=1e-9)
        assert np.isclose(target.dec.deg, 45.678, rtol=1e-9)

    def test_resolve_decimal_degrees_comma_no_spaces(self):
        """Test resolving comma-separated without spaces."""
        target = resolve.resolve("123.456,45.678")
        assert target is not None
        assert np.isclose(target.ra.deg, 123.456, rtol=1e-9)
        assert np.isclose(target.dec.deg, 45.678, rtol=1e-9)


class TestResolveSexagesimal:
    """Test resolve() with sexagesimal coordinates (no network)."""

    def test_resolve_sexagesimal_space_separated(self):
        """Test resolving space-separated sexagesimal coordinates."""
        # RA: 12h 34m 56s = 12*15 + 34/60*15 + 56/3600*15 = 188.733333 deg
        # Dec: +45d 12m 34s = 45 + 12/60 + 34/3600 = 45.209444 deg
        target = resolve.resolve("12 34 56 +45 12 34")
        assert target is not None
        expected_ra = (12 + 34/60 + 56/3600) * 15
        expected_dec = 45 + 12/60 + 34/3600
        assert np.isclose(target.ra.deg, expected_ra, rtol=1e-6)
        assert np.isclose(target.dec.deg, expected_dec, rtol=1e-6)

    def test_resolve_sexagesimal_colon_separated(self):
        """Test resolving colon-separated sexagesimal coordinates."""
        target = resolve.resolve("12:34:56 +45:12:34")
        assert target is not None
        expected_ra = (12 + 34/60 + 56/3600) * 15
        expected_dec = 45 + 12/60 + 34/3600
        assert np.isclose(target.ra.deg, expected_ra, rtol=1e-6)
        assert np.isclose(target.dec.deg, expected_dec, rtol=1e-6)

    def test_resolve_sexagesimal_negative_dec(self):
        """Test resolving sexagesimal with negative declination."""
        target = resolve.resolve("12 34 56 -45 12 34")
        assert target is not None
        expected_ra = (12 + 34/60 + 56/3600) * 15
        expected_dec = -(45 + 12/60 + 34/3600)
        assert np.isclose(target.ra.deg, expected_ra, rtol=1e-6)
        assert np.isclose(target.dec.deg, expected_dec, rtol=1e-6)

    def test_resolve_sexagesimal_no_sign(self):
        """Test resolving sexagesimal without explicit sign (positive)."""
        target = resolve.resolve("12 34 56 45 12 34")
        assert target is not None
        expected_ra = (12 + 34/60 + 56/3600) * 15
        expected_dec = 45 + 12/60 + 34/3600
        assert np.isclose(target.ra.deg, expected_ra, rtol=1e-6)
        assert np.isclose(target.dec.deg, expected_dec, rtol=1e-6)

    def test_resolve_sexagesimal_zero_ra_dec(self):
        """Test resolving zero sexagesimal coordinates."""
        target = resolve.resolve("0 0 0 0 0 0")
        assert target is not None
        assert np.isclose(target.ra.deg, 0.0, rtol=1e-9)
        assert np.isclose(target.dec.deg, 0.0, rtol=1e-9)

    def test_resolve_sexagesimal_with_decimals(self):
        """Test resolving sexagesimal with decimal seconds."""
        target = resolve.resolve("12 34 56.789 +45 12 34.567")
        assert target is not None
        expected_ra = (12 + 34/60 + 56.789/3600) * 15
        expected_dec = 45 + 12/60 + 34.567/3600
        assert np.isclose(target.ra.deg, expected_ra, rtol=1e-6)
        assert np.isclose(target.dec.deg, expected_dec, rtol=1e-6)

    def test_resolve_sexagesimal_space_comma_separated(self):
        """Test resolving comma-separated sexagesimal (space format)."""
        target = resolve.resolve("12 34 56, +45 12 34")
        assert target is not None
        expected_ra = (12 + 34/60 + 56/3600) * 15
        expected_dec = 45 + 12/60 + 34/3600
        assert np.isclose(target.ra.deg, expected_ra, rtol=1e-6)
        assert np.isclose(target.dec.deg, expected_dec, rtol=1e-6)

    def test_resolve_sexagesimal_colon_comma_separated(self):
        """Test resolving comma-separated sexagesimal (colon format)."""
        target = resolve.resolve("12:34:56, +45:12:34")
        assert target is not None
        expected_ra = (12 + 34/60 + 56/3600) * 15
        expected_dec = 45 + 12/60 + 34/3600
        assert np.isclose(target.ra.deg, expected_ra, rtol=1e-6)
        assert np.isclose(target.dec.deg, expected_dec, rtol=1e-6)

    def test_resolve_sexagesimal_comma_negative_dec(self):
        """Test resolving comma-separated sexagesimal with negative Dec."""
        target = resolve.resolve("12 34 56, -45 12 34")
        assert target is not None
        expected_ra = (12 + 34/60 + 56/3600) * 15
        expected_dec = -(45 + 12/60 + 34/3600)
        assert np.isclose(target.ra.deg, expected_ra, rtol=1e-6)
        assert np.isclose(target.dec.deg, expected_dec, rtol=1e-6)

    def test_resolve_sexagesimal_comma_no_spaces(self):
        """Test resolving comma-separated sexagesimal without spaces around comma."""
        target = resolve.resolve("12 34 56,+45 12 34")
        assert target is not None
        expected_ra = (12 + 34/60 + 56/3600) * 15
        expected_dec = 45 + 12/60 + 34/3600
        assert np.isclose(target.ra.deg, expected_ra, rtol=1e-6)
        assert np.isclose(target.dec.deg, expected_dec, rtol=1e-6)


class TestSimbadResolve:
    """Test simbadResolve() with mocked network calls."""

    @patch('stdpipe.resolve.requests.get')
    def test_simbad_resolve_success(self, mock_get):
        """Test successful Simbad resolution."""
        # Mock XML response from Simbad
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <Sesame>
            <Resolver>
                <oname>M 31</oname>
                <jradeg>10.6847083</jradeg>
                <jdedeg>41.2687500</jdedeg>
            </Resolver>
        </Sesame>
        """
        mock_response = Mock()
        mock_response.content = mock_xml.encode('utf-8')
        mock_get.return_value = mock_response

        name, ra, dec = resolve.simbadResolve('M31')

        assert name == 'M 31'
        assert np.isclose(ra, 10.6847083, rtol=1e-9)
        assert np.isclose(dec, 41.2687500, rtol=1e-9)
        mock_get.assert_called_once()

    @patch('stdpipe.resolve.requests.get')
    def test_simbad_resolve_failure(self, mock_get):
        """Test Simbad resolution failure (invalid XML)."""
        mock_response = Mock()
        mock_response.content = b"Invalid XML"
        mock_get.return_value = mock_response

        name, ra, dec = resolve.simbadResolve('InvalidObject')

        assert name is None
        assert ra is None
        assert dec is None

    @patch('stdpipe.resolve.requests.get')
    def test_simbad_resolve_network_error(self, mock_get):
        """Test Simbad resolution with network error.

        Note: Network errors at the requests.get() level are NOT caught by
        simbadResolve(), so the exception propagates to the caller.
        """
        mock_get.side_effect = Exception("Network error")

        # Network errors are not caught by simbadResolve - they propagate
        with pytest.raises(Exception, match="Network error"):
            resolve.simbadResolve('M31')

    @patch('stdpipe.resolve.requests.get')
    def test_simbad_resolve_missing_fields(self, mock_get):
        """Test Simbad resolution with incomplete XML."""
        # Missing jdedeg field
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <Sesame>
            <Resolver>
                <oname>M 31</oname>
                <jradeg>10.6847083</jradeg>
            </Resolver>
        </Sesame>
        """
        mock_response = Mock()
        mock_response.content = mock_xml.encode('utf-8')
        mock_get.return_value = mock_response

        name, ra, dec = resolve.simbadResolve('M31')

        assert name is None
        assert ra is None
        assert dec is None


class TestTnsResolve:
    """Test tnsResolve() with mocked network calls."""

    @patch('stdpipe.resolve.requests.post')
    def test_tns_resolve_success(self, mock_post):
        """Test successful TNS resolution."""
        mock_response = Mock()
        mock_response.content = json.dumps([{
            'd:fullname': 'AT 2023lxx',
            'd:ra': 123.456,
            'd:declination': -45.678
        }]).encode('utf-8')
        mock_post.return_value = mock_response

        name, ra, dec = resolve.tnsResolve('AT2023lxx')

        assert name == 'AT 2023lxx'
        assert np.isclose(ra, 123.456, rtol=1e-9)
        assert np.isclose(dec, -45.678, rtol=1e-9)

    @patch('stdpipe.resolve.requests.post')
    def test_tns_resolve_with_space(self, mock_post):
        """Test TNS resolution with space in name."""
        mock_response = Mock()
        mock_response.content = json.dumps([{
            'd:fullname': 'AT 2023lxx',
            'd:ra': 123.456,
            'd:declination': -45.678
        }]).encode('utf-8')
        mock_post.return_value = mock_response

        name, ra, dec = resolve.tnsResolve('AT 2023lxx')

        assert name == 'AT 2023lxx'
        assert np.isclose(ra, 123.456, rtol=1e-9)
        assert np.isclose(dec, -45.678, rtol=1e-9)

    @patch('stdpipe.resolve.requests.post')
    def test_tns_resolve_normalizes_name(self, mock_post):
        """Test that TNS resolver normalizes AT names (removes space, adds it back)."""
        mock_response = Mock()
        mock_response.content = json.dumps([{
            'd:fullname': 'AT 2023lxx',
            'd:ra': 123.456,
            'd:declination': -45.678
        }]).encode('utf-8')
        mock_post.return_value = mock_response

        # Input without space
        name, ra, dec = resolve.tnsResolve('AT2023lxx')

        # Should call API with space
        assert mock_post.call_count >= 1
        first_call_args = mock_post.call_args_list[0]
        assert first_call_args[1]['json']['name'] == 'AT 2023lxx'

    @patch('stdpipe.resolve.requests.post')
    def test_tns_resolve_empty_response(self, mock_post):
        """Test TNS resolution with empty response."""
        mock_response = Mock()
        mock_response.content = json.dumps([]).encode('utf-8')
        mock_post.return_value = mock_response

        name, ra, dec = resolve.tnsResolve('AT2023xxx')

        assert name is None
        assert ra is None
        assert dec is None

    @patch('stdpipe.resolve.requests.post')
    def test_tns_resolve_network_error(self, mock_post):
        """Test TNS resolution with network error.

        Note: Network errors at the requests.post() level are NOT caught by
        tnsResolve(), so the exception propagates to the caller.
        """
        mock_post.side_effect = Exception("Network error")

        # Network errors are not caught by tnsResolve - they propagate
        with pytest.raises(Exception, match="Network error"):
            resolve.tnsResolve('AT2023lxx')

    @patch('stdpipe.resolve.requests.post')
    def test_tns_resolve_tries_reverse(self, mock_post):
        """Test that TNS resolver tries reverse lookup if first attempt fails."""
        # First call returns empty, second call returns result
        mock_response_empty = Mock()
        mock_response_empty.content = json.dumps([]).encode('utf-8')

        mock_response_success = Mock()
        mock_response_success.content = json.dumps([{
            'd:fullname': 'AT 2023lxx',
            'd:ra': 123.456,
            'd:declination': -45.678
        }]).encode('utf-8')

        mock_post.side_effect = [mock_response_empty, mock_response_success]

        name, ra, dec = resolve.tnsResolve('AT2023lxx')

        assert name == 'AT 2023lxx'
        assert mock_post.call_count == 2
        # Second call should have reverse=True
        second_call_args = mock_post.call_args_list[1]
        assert second_call_args[1]['json']['reverse'] is True


class TestResolveWithMocks:
    """Test resolve() main function with mocked network calls."""

    @patch('stdpipe.resolve.simbadResolve')
    def test_resolve_falls_back_to_simbad(self, mock_simbad):
        """Test that resolve() falls back to Simbad for object names."""
        mock_simbad.return_value = ('M 31', 10.6847083, 41.2687500)

        target = resolve.resolve('M31')

        assert target is not None
        assert np.isclose(target.ra.deg, 10.6847083, rtol=1e-6)
        assert np.isclose(target.dec.deg, 41.2687500, rtol=1e-6)
        mock_simbad.assert_called_once_with('M31')

    @patch('stdpipe.resolve.tnsResolve')
    @patch('stdpipe.resolve.simbadResolve')
    def test_resolve_falls_back_to_tns(self, mock_simbad, mock_tns):
        """Test that resolve() falls back to TNS if Simbad fails."""
        mock_simbad.return_value = (None, None, None)
        mock_tns.return_value = ('AT 2023lxx', 123.456, -45.678)

        target = resolve.resolve('AT2023lxx')

        assert target is not None
        assert np.isclose(target.ra.deg, 123.456, rtol=1e-6)
        assert np.isclose(target.dec.deg, -45.678, rtol=1e-6)
        mock_simbad.assert_called_once()
        mock_tns.assert_called_once()

    @patch('stdpipe.resolve.SkyCoord.from_name')
    @patch('stdpipe.resolve.tnsResolve')
    @patch('stdpipe.resolve.simbadResolve')
    def test_resolve_falls_back_to_astropy(self, mock_simbad, mock_tns, mock_from_name):
        """Test that resolve() falls back to Astropy if Simbad and TNS fail."""
        mock_simbad.return_value = (None, None, None)
        mock_tns.return_value = (None, None, None)
        mock_from_name.return_value = SkyCoord(ra=234.567, dec=56.789, unit='deg')

        target = resolve.resolve('NGC1234')

        assert target is not None
        assert np.isclose(target.ra.deg, 234.567, rtol=1e-6)
        assert np.isclose(target.dec.deg, 56.789, rtol=1e-6)
        mock_simbad.assert_called_once()
        mock_tns.assert_called_once()
        mock_from_name.assert_called_once()

    @patch('stdpipe.resolve.SkyCoord.from_name')
    @patch('stdpipe.resolve.tnsResolve')
    @patch('stdpipe.resolve.simbadResolve')
    def test_resolve_returns_none_if_all_fail(self, mock_simbad, mock_tns, mock_from_name):
        """Test that resolve() returns None if all methods fail."""
        mock_simbad.return_value = (None, None, None)
        mock_tns.return_value = (None, None, None)
        mock_from_name.side_effect = Exception("Not found")

        target = resolve.resolve('InvalidObject123')

        assert target is None

    def test_resolve_prefers_coordinate_parsing_over_network(self):
        """Test that resolve() tries coordinate parsing before network calls."""
        # No mocking needed - coordinate parsing should work without network
        target = resolve.resolve("123.456 45.678")

        assert target is not None
        assert np.isclose(target.ra.deg, 123.456, rtol=1e-9)
        assert np.isclose(target.dec.deg, 45.678, rtol=1e-9)

    @patch('stdpipe.resolve.simbadResolve')
    def test_resolve_verbose_mode(self, mock_simbad, capsys):
        """Test resolve() verbose output."""
        mock_simbad.return_value = ('M 31', 10.6847083, 41.2687500)

        target = resolve.resolve('M31', verbose=True)

        captured = capsys.readouterr()
        assert 'Resolved by Simbad' in captured.out
        assert 'M 31' in captured.out
        assert 'RA =' in captured.out
        assert 'Dec =' in captured.out

    @patch('stdpipe.resolve.SkyCoord.from_name')
    @patch('stdpipe.resolve.tnsResolve')
    @patch('stdpipe.resolve.simbadResolve')
    def test_resolve_verbose_mode_failure(self, mock_simbad, mock_tns, mock_from_name, capsys):
        """Test resolve() verbose output on failure."""
        mock_simbad.return_value = (None, None, None)
        mock_tns.return_value = (None, None, None)
        mock_from_name.side_effect = Exception("Not found")

        target = resolve.resolve('InvalidObject', verbose=True)

        captured = capsys.readouterr()
        assert 'Failed to resolve' in captured.out
        assert 'InvalidObject' in captured.out

    def test_resolve_verbose_with_callable(self):
        """Test resolve() with callable verbose function."""
        messages = []

        def log_func(*args, **kwargs):
            messages.append(' '.join(str(arg) for arg in args))

        target = resolve.resolve("123.456 45.678", verbose=log_func)

        assert target is not None
        assert len(messages) > 0
        assert any('degrees' in msg for msg in messages)


# Import json for TNS tests
import json


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
