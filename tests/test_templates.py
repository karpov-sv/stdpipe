"""
Unit tests for templates.py — template image retrieval and processing utilities.

Tests pure functions and mockable code paths without requiring network access.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from astropy.io import fits
from astropy.table import Table, MaskedColumn
from astropy.stats import mad_std

from stdpipe import templates


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def ps1_header_old_wcs():
    """FITS header with old-style PS1 WCS keywords (PC001001, no RADESYS)."""
    h = fits.Header()
    h['NAXIS'] = 2
    h['NAXIS1'] = 50
    h['NAXIS2'] = 50
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CRPIX1'] = 25.0
    h['CRPIX2'] = 25.0
    h['CRVAL1'] = 180.0
    h['CRVAL2'] = 45.0
    h['PC001001'] = 1.0
    h['PC001002'] = 0.0
    h['PC002001'] = 0.0
    h['PC002002'] = 1.0
    return h


@pytest.fixture
def ps1_header_asinh(ps1_header_old_wcs):
    """PS1 header with asinh scaling keywords."""
    h = ps1_header_old_wcs.copy()
    h['BSOFTEN'] = 2.0
    h['BOFFSET'] = 0.5
    h['EXPTIME'] = 30.0
    h['BLANK'] = -9999
    return h


@pytest.fixture
def catalog_on_image(simple_wcs):
    """Catalog with stars on the image footprint.

    Uses simple_wcs from conftest (256x256, 1 arcsec/pix, center RA=180 Dec=45).
    """
    # Pixel (128, 128) → center of image
    # Pixel (100, 100) → offset from center
    ra_center, dec_center = simple_wcs.all_pix2world(128, 128, 0)
    ra_off, dec_off = simple_wcs.all_pix2world(100, 100, 0)

    cat = Table()
    cat['RAJ2000'] = MaskedColumn([float(ra_center), float(ra_off)])
    cat['DEJ2000'] = MaskedColumn([float(dec_center), float(dec_off)])
    cat['rmag'] = MaskedColumn([10.0, 16.0])  # First is bright (saturated)
    cat['e_rmag'] = MaskedColumn([0.01, 0.02])
    return cat


def _make_mock_hdu(image, header=None):
    """Create a mock HDU list for patching fits.open."""
    if header is None:
        header = fits.Header()
    hdu = MagicMock()
    hdu[0].data = image
    hdu[0].header = header
    hdu.close = MagicMock()
    return hdu


# ========================================================================
# TestDilateMask
# ========================================================================


class TestDilateMask:
    """Test mask dilation via morphological Tophat kernel."""

    @pytest.mark.unit
    def test_all_false_stays_false(self):
        """Empty mask should remain empty after dilation."""
        mask = np.zeros((50, 50), dtype=bool)
        result = templates.dilate_mask(mask, dilate=5)
        assert not np.any(result)

    @pytest.mark.unit
    def test_single_pixel_expands(self):
        """Single True pixel should expand to a disk."""
        mask = np.zeros((51, 51), dtype=bool)
        mask[25, 25] = True
        result = templates.dilate_mask(mask, dilate=5)

        assert result.dtype == bool
        # Center should be True
        assert result[25, 25]
        # Pixels within radius should be True
        assert result[25, 30]  # 5 pixels right
        assert result[25, 20]  # 5 pixels left
        # Pixels well outside should be False
        assert not result[25, 40]
        assert not result[0, 0]
        # Should have expanded (more than just 1 pixel)
        assert np.sum(result) > 1

    @pytest.mark.unit
    def test_all_true_stays_true(self):
        """Fully True mask should remain fully True."""
        mask = np.ones((50, 50), dtype=bool)
        result = templates.dilate_mask(mask, dilate=5)
        assert np.all(result)

    @pytest.mark.unit
    def test_dilate_1_minimal(self):
        """dilate=1 should produce minimal expansion (~3x3 disk)."""
        mask = np.zeros((21, 21), dtype=bool)
        mask[10, 10] = True
        result = templates.dilate_mask(mask, dilate=1)

        # Center and immediate neighbors should be True
        assert result[10, 10]
        assert result[10, 11]
        assert result[11, 10]
        # Pixels 3+ away should be False
        assert not result[10, 14]
        assert not result[14, 10]

    @pytest.mark.unit
    def test_return_dtype_is_bool(self):
        """Result should be boolean."""
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10] = True
        result = templates.dilate_mask(mask, dilate=3)
        assert result.dtype == bool


# ========================================================================
# TestNormalizePS1
# ========================================================================


class TestNormalizePS1:
    """Test Pan-STARRS skycell normalization."""

    @pytest.mark.unit
    def test_wcs_keyword_renaming(self, ps1_header_old_wcs):
        """Old-style PC001001 keywords should be renamed to PC1_1 format."""
        image = np.ones((50, 50))
        _, header_out = templates.normalize_ps1_skycell(image, ps1_header_old_wcs)

        assert 'RADESYS' in header_out
        assert header_out['RADESYS'] == 'FK5'
        assert 'PC1_1' in header_out
        assert 'PC1_2' in header_out
        assert 'PC2_1' in header_out
        assert 'PC2_2' in header_out
        assert header_out['PC1_1'] == 1.0
        assert header_out['PC2_2'] == 1.0
        # Old keywords should be gone (renamed, not copied)
        assert 'PC001001' not in header_out

    @pytest.mark.unit
    def test_asinh_linearization(self, ps1_header_asinh):
        """Asinh scaling should be linearized using BSOFTEN, BOFFSET, EXPTIME."""
        image = np.array([[0.0, 1.0], [2.0, -1.0]])

        image_out, header_out = templates.normalize_ps1_skycell(
            image.copy(), ps1_header_asinh
        )

        bsoften = ps1_header_asinh['BSOFTEN']
        boffset = ps1_header_asinh['BOFFSET']
        exptime = ps1_header_asinh['EXPTIME']

        # Compute expected values manually
        for iy in range(2):
            for ix in range(2):
                x = image[iy, ix] * 0.4 * np.log(10)
                expected = (boffset + bsoften * (np.exp(x) - np.exp(-x))) / exptime
                np.testing.assert_allclose(
                    image_out[iy, ix], expected, rtol=1e-10,
                    err_msg=f"Pixel ({iy},{ix}): expected {expected}, got {image_out[iy, ix]}"
                )

        # BSOFTEN, BOFFSET, BLANK should be removed
        assert 'BSOFTEN' not in header_out
        assert 'BOFFSET' not in header_out
        assert 'BLANK' not in header_out

    @pytest.mark.unit
    def test_already_has_radesys_unchanged(self):
        """Header with RADESYS should pass through unchanged."""
        h = fits.Header()
        h['RADESYS'] = 'ICRS'
        h['PC001001'] = 1.0  # Present but should NOT be renamed
        image = np.ones((10, 10)) * 42.0

        image_out, header_out = templates.normalize_ps1_skycell(image.copy(), h)

        # Image unchanged
        np.testing.assert_array_equal(image_out, 42.0)
        # Header unchanged — PC001001 still there (rename only happens when RADESYS absent)
        assert 'PC001001' in header_out

    @pytest.mark.unit
    def test_original_header_not_mutated(self, ps1_header_old_wcs):
        """The original header should not be modified."""
        original_keys = set(ps1_header_old_wcs.keys())
        image = np.ones((50, 50))

        templates.normalize_ps1_skycell(image, ps1_header_old_wcs)

        # Original should still have old keywords
        assert 'PC001001' in ps1_header_old_wcs
        assert 'RADESYS' not in ps1_header_old_wcs
        assert set(ps1_header_old_wcs.keys()) == original_keys

    @pytest.mark.unit
    def test_missing_blank_no_crash(self, ps1_header_old_wcs):
        """Header without BLANK keyword should not crash during removal."""
        # Add BSOFTEN/BOFFSET/EXPTIME but no BLANK
        h = ps1_header_old_wcs.copy()
        h['BSOFTEN'] = 1.0
        h['BOFFSET'] = 0.0
        h['EXPTIME'] = 1.0
        assert 'BLANK' not in h

        image = np.zeros((50, 50))
        # Should not raise
        image_out, header_out = templates.normalize_ps1_skycell(image, h)
        assert 'BLANK' not in header_out


# ========================================================================
# TestPointInSurvey
# ========================================================================


class TestPointInSurvey:
    """Test survey footprint checks."""

    @pytest.mark.unit
    def test_point_in_ps1_above_limit(self):
        """Declination > -30 should be in PS1."""
        assert templates.point_in_ps1(0, 0) is True
        assert templates.point_in_ps1(180, 45) is True
        assert templates.point_in_ps1(0, 90) is True

    @pytest.mark.unit
    def test_point_in_ps1_below_limit(self):
        """Declination <= -30 should not be in PS1."""
        assert templates.point_in_ps1(0, -31) is False
        # Boundary: -30 is NOT included (strict >)
        assert templates.point_in_ps1(0, -30) is False

    @pytest.mark.unit
    def test_point_in_ps1_array(self):
        """Array inputs should work."""
        dec = np.array([-40, -30, -29, 0, 45])
        result = templates.point_in_ps1(0, dec)
        expected = np.array([False, False, True, True, True])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_point_in_ls_high_latitude(self):
        """Points far from Galactic plane (|b| > 20) should be in LS."""
        # North Galactic Pole: RA~192.86, Dec~27.13 → b≈90
        assert templates.point_in_ls(192.86, 27.13) == True

    @pytest.mark.unit
    def test_point_in_ls_galactic_center(self):
        """Points near Galactic plane (|b| < 20) should not be in LS."""
        # Galactic center: RA~266.4, Dec~-29.0 → b≈0
        assert templates.point_in_ls(266.4, -29.0) == False

    @pytest.mark.unit
    def test_point_in_ls_array(self):
        """Array inputs should return array."""
        ra = np.array([192.86, 266.4])
        dec = np.array([27.13, -29.0])
        result = templates.point_in_ls(ra, dec)
        assert len(result) == 2
        assert result[0] == True   # High |b|
        assert result[1] == False  # Low |b|


# ========================================================================
# TestGetHipsImage
# ========================================================================


class TestGetHipsImage:
    """Test HiPS image retrieval — parameter validation and post-processing."""

    @pytest.mark.unit
    def test_missing_coordinates_returns_none(self):
        """No WCS and no ra/dec/fov should return (None, None)."""
        result = templates.get_hips_image(
            'CDS/P/DSS2/red', width=100, height=100
        )
        assert result == (None, None)

    @pytest.mark.unit
    def test_missing_coordinates_no_header_returns_none(self):
        """get_header=False on missing-coordinates path should return None."""
        result = templates.get_hips_image(
            'CDS/P/DSS2/red', width=100, height=100, get_header=False
        )
        # The function returns None, None even with get_header=False on this path
        # (it returns before checking get_header)
        assert result == (None, None)

    @pytest.mark.unit
    @patch('stdpipe.templates.fits.open')
    def test_panstarrs_auto_asinh(self, mock_fits_open):
        """PanSTARRS non-g band should auto-detect asinh scaling."""
        raw_value = 1.0
        image = np.full((20, 20), raw_value)
        mock_fits_open.return_value = _make_mock_hdu(image.copy())

        result, _ = templates.get_hips_image(
            'PanSTARRS/DR1/r',
            ra=180, dec=45, fov=0.01,
            width=20, height=20,
            normalize=False,
        )

        # asinh linearization: x = val * 0.4 * log(10), result = exp(x) - exp(-x)
        x = raw_value * 0.4 * np.log(10)
        expected = np.exp(x) - np.exp(-x)
        np.testing.assert_allclose(result[10, 10], expected, rtol=1e-10)

    @pytest.mark.unit
    @patch('stdpipe.templates.fits.open')
    def test_asinh_false_override(self, mock_fits_open):
        """Explicit asinh=False should skip linearization even for PanSTARRS."""
        raw_value = 1.0
        image = np.full((20, 20), raw_value)
        mock_fits_open.return_value = _make_mock_hdu(image.copy())

        result, _ = templates.get_hips_image(
            'PanSTARRS/DR1/r',
            ra=180, dec=45, fov=0.01,
            width=20, height=20,
            asinh=False,
            normalize=False,
        )

        # Should NOT have applied asinh linearization
        # But normalize=False means no normalization either
        # The .astype(np.double) copy means the values should be unchanged
        np.testing.assert_allclose(result[10, 10], raw_value, rtol=1e-10)

    @pytest.mark.unit
    @patch('stdpipe.templates.fits.open')
    def test_normalize_rescales(self, mock_fits_open):
        """normalize=True should produce median≈100 and mad_std≈10."""
        rng = np.random.RandomState(42)
        image = rng.normal(500, 20, (50, 50))
        mock_fits_open.return_value = _make_mock_hdu(image.copy())

        result, _ = templates.get_hips_image(
            'CDS/P/DSS2/red',
            ra=180, dec=45, fov=0.01,
            width=50, height=50,
            normalize=True,
        )

        assert abs(np.nanmedian(result) - 100) < 1.0
        assert abs(mad_std(result, ignore_nan=True) - 10) < 1.0

    @pytest.mark.unit
    @patch('stdpipe.templates.fits.open')
    def test_panstarrs_g_no_asinh(self, mock_fits_open):
        """PanSTARRS g-band should NOT auto-detect asinh."""
        raw_value = 1.0
        image = np.full((20, 20), raw_value)
        mock_fits_open.return_value = _make_mock_hdu(image.copy())

        result, _ = templates.get_hips_image(
            'PanSTARRS/DR1/g',
            ra=180, dec=45, fov=0.01,
            width=20, height=20,
            normalize=False,
        )

        # g-band: no asinh, no normalize → values unchanged
        np.testing.assert_allclose(result[10, 10], raw_value, rtol=1e-10)

    @pytest.mark.unit
    @patch('stdpipe.templates.fits.open')
    def test_get_header_false(self, mock_fits_open):
        """get_header=False should return just the image."""
        image = np.ones((20, 20))
        mock_fits_open.return_value = _make_mock_hdu(image.copy())

        result = templates.get_hips_image(
            'CDS/P/DSS2/red',
            ra=180, dec=45, fov=0.01,
            width=20, height=20,
            normalize=False,
            get_header=False,
        )

        # Should be a single array, not a tuple
        assert isinstance(result, np.ndarray)
        assert result.shape == (20, 20)

    @pytest.mark.unit
    @patch('stdpipe.templates.fits.open')
    def test_download_failure_returns_none(self, mock_fits_open):
        """If all download attempts fail, should return (None, None)."""
        mock_fits_open.side_effect = Exception("Connection refused")

        result = templates.get_hips_image(
            'CDS/P/DSS2/red',
            ra=180, dec=45, fov=0.01,
            width=20, height=20,
        )

        assert result == (None, None)

    @pytest.mark.unit
    @patch('stdpipe.templates.fits.open')
    def test_download_failure_get_header_false(self, mock_fits_open):
        """Download failure with get_header=False should return None."""
        mock_fits_open.side_effect = Exception("Connection refused")

        result = templates.get_hips_image(
            'CDS/P/DSS2/red',
            ra=180, dec=45, fov=0.01,
            width=20, height=20,
            get_header=False,
        )

        assert result is None


# ========================================================================
# TestMaskTemplate
# ========================================================================


class TestMaskTemplate:
    """Test template masking heuristics."""

    @pytest.mark.unit
    def test_mask_nans_true(self):
        """NaN pixels should be masked when mask_nans=True."""
        image = np.ones((50, 50))
        image[10, 15] = np.nan
        image[20, 25] = np.inf  # inf is finite=False → should also be masked
        image[30, 35] = -np.inf

        mask = templates.mask_template(image, mask_nans=True, dilate=0)

        assert mask[10, 15]   # NaN → masked
        assert mask[20, 25]   # inf → masked
        assert mask[30, 35]   # -inf → masked
        assert not mask[0, 0] # Normal pixel → not masked

    @pytest.mark.unit
    def test_mask_nans_false(self):
        """No NaN masking when mask_nans=False."""
        image = np.ones((50, 50))
        image[10, 15] = np.nan

        mask = templates.mask_template(image, mask_nans=False, dilate=0)

        assert not np.any(mask)

    @pytest.mark.unit
    def test_no_catalog(self):
        """Without catalog, only NaN masking should happen."""
        image = np.ones((50, 50))
        image[5, 5] = np.nan

        mask = templates.mask_template(image, cat=None, mask_nans=True, dilate=0)

        assert mask[5, 5]
        assert np.sum(mask) == 1  # Only the NaN pixel

    @pytest.mark.unit
    def test_dilate_zero_skipped(self):
        """dilate=0 should not expand the mask."""
        image = np.ones((50, 50))
        image[25, 25] = np.nan

        mask = templates.mask_template(image, mask_nans=True, dilate=0)

        assert mask[25, 25]
        assert np.sum(mask) == 1  # No expansion

    @pytest.mark.unit
    def test_dilate_expands_mask(self):
        """Non-zero dilate should expand the NaN mask."""
        image = np.ones((50, 50))
        image[25, 25] = np.nan

        mask = templates.mask_template(image, mask_nans=True, dilate=3)

        assert mask[25, 25]
        # Dilation should have expanded the mask
        assert np.sum(mask) > 1
        # Nearby pixels should be masked
        assert mask[25, 26]
        assert mask[26, 25]

    @pytest.mark.unit
    def test_catalog_saturation_masking(self, simple_wcs, catalog_on_image):
        """Bright catalog stars should be masked at their center pixel."""
        image = np.ones((256, 256))

        mask = templates.mask_template(
            image,
            cat=catalog_on_image,
            cat_saturation_mag=12.0,  # First star (mag=10) is below this
            wcs=simple_wcs,
            mask_nans=False,
            mask_masked=False,
            dilate=0,
        )

        # First star (mag=10 < 12) should be masked at its center pixel
        # Second star (mag=16 > 12) should NOT be masked
        assert np.sum(mask) == 1

    @pytest.mark.unit
    def test_catalog_saturation_mag_none_no_crash(self, simple_wcs, catalog_on_image):
        """cat_saturation_mag=None with catalog should not crash."""
        image = np.ones((256, 256))

        # This previously crashed with TypeError on the log format string
        mask = templates.mask_template(
            image,
            cat=catalog_on_image,
            cat_saturation_mag=None,
            wcs=simple_wcs,
            mask_nans=False,
            mask_masked=False,
            dilate=0,
        )

        assert mask.dtype == bool

    @pytest.mark.unit
    def test_catalog_non_masked_column(self, simple_wcs):
        """Catalog with plain (non-masked) columns should not crash."""
        image = np.ones((256, 256))

        ra1, dec1 = simple_wcs.all_pix2world(128, 128, 0)
        cat = Table()
        cat['RAJ2000'] = [float(ra1)]
        cat['DEJ2000'] = [float(dec1)]
        cat['rmag'] = [14.0]       # Plain Column, no .mask attribute
        cat['e_rmag'] = [0.01]

        # This previously crashed with AttributeError on .mask
        mask = templates.mask_template(
            image,
            cat=cat,
            wcs=simple_wcs,
            mask_nans=False,
            mask_masked=True,
            cat_saturation_mag=0.0,
            dilate=0,
        )

        assert mask.dtype == bool

    @pytest.mark.unit
    def test_catalog_masked_column_masking(self, simple_wcs):
        """Stars with masked magnitudes should be masked."""
        image = np.ones((256, 256))

        # Create catalog where second star has masked magnitude
        ra1, dec1 = simple_wcs.all_pix2world(128, 128, 0)
        ra2, dec2 = simple_wcs.all_pix2world(100, 100, 0)

        cat = Table()
        cat['RAJ2000'] = MaskedColumn([float(ra1), float(ra2)])
        cat['DEJ2000'] = MaskedColumn([float(dec1), float(dec2)])
        cat['rmag'] = MaskedColumn(
            [14.0, 15.0], mask=[False, True]  # Second star's mag is masked
        )
        cat['e_rmag'] = MaskedColumn([0.01, 0.02])

        mask = templates.mask_template(
            image,
            cat=cat,
            wcs=simple_wcs,
            mask_nans=False,
            mask_masked=True,
            cat_saturation_mag=0.0,  # Set to 0 so no star is "saturated" — we test mask_masked only
            dilate=0,
        )

        # Second star (masked mag) should be masked
        assert np.sum(mask) == 1

    @pytest.mark.unit
    def test_verbose_logging(self):
        """Verbose callable should receive log messages."""
        image = np.ones((50, 50))
        image[25, 25] = np.nan
        log_messages = []

        templates.mask_template(
            image, mask_nans=True, dilate=0,
            verbose=lambda *args: log_messages.append(args)
        )

        # Should have logged at least one message about NaN masking
        assert len(log_messages) > 0


# ========================================================================
# Skycell footprint filtering tests
# ========================================================================


def _make_wcs(ra0=180.0, dec0=45.0, cdelt=1.0/3600, nx=256, ny=256):
    """Build a simple TAN WCS for testing."""
    from astropy.wcs import WCS

    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL1'] = ra0
    header['CRVAL2'] = dec0
    header['CRPIX1'] = nx / 2 + 0.5
    header['CRPIX2'] = ny / 2 + 0.5
    header['CD1_1'] = -cdelt
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = cdelt
    header['EQUINOX'] = 2000.0
    return WCS(header)


@pytest.mark.unit
class TestFilterCellsByFootprint:

    def test_cell_inside_kept(self):
        """Cell at image centre should be kept."""
        wcs = _make_wcs()
        keep = templates._filter_cells_by_footprint(
            np.array([180.0]), np.array([45.0]),
            cell_radius=0.01, wcs=wcs, width=256, height=256,
        )
        assert keep[0]

    def test_cell_outside_rejected(self):
        """Cell far from image should be rejected."""
        wcs = _make_wcs()
        keep = templates._filter_cells_by_footprint(
            np.array([185.0]), np.array([45.0]),
            cell_radius=0.01, wcs=wcs, width=256, height=256,
        )
        assert not keep[0]

    def test_cell_near_edge_kept(self):
        """Cell just outside image edge but within cell_radius should be kept."""
        wcs = _make_wcs()
        # Image half-width in Dec: 128 px * (1/3600) deg = 0.0356 deg
        # Place cell 0.04 deg away in Dec (just outside) with cell_radius=0.01 (36 px margin)
        keep = templates._filter_cells_by_footprint(
            np.array([180.0]),
            np.array([45.0 + 0.04]),
            cell_radius=0.01, wcs=wcs, width=256, height=256,
        )
        assert keep[0]

    def test_elongated_field_rejects_corner_cells(self):
        """For an elongated field, cells near the bounding circle corners
        but outside the rectangle should be rejected."""
        # 1024 x 128 image: very elongated
        wcs = _make_wcs(nx=1024, ny=128)
        # The circumscribed circle has radius ~sqrt(512^2+64^2)*cdelt ~0.143 deg
        # A cell at (ra0+0.12, dec0+0.03) is within that circle but
        # outside the narrow height extent (~0.018 deg from centre)
        offset_ra = 0.12 / np.cos(np.radians(45))
        keep = templates._filter_cells_by_footprint(
            np.array([180.0 - offset_ra]),
            np.array([45.03]),
            cell_radius=0.003, wcs=wcs, width=1024, height=128,
        )
        assert not keep[0]

    def test_multiple_cells_mixed(self):
        """Mix of inside, edge and outside cells."""
        wcs = _make_wcs()
        # Half-width in Dec: 0.0356 deg; in RA: 0.0356/cos(45)=0.0503 deg
        ras = np.array([180.0, 180.0, 185.0])
        decs = np.array([45.0, 45.1, 45.0])  # 0.1 deg in Dec is well outside
        keep = templates._filter_cells_by_footprint(
            ras, decs, cell_radius=0.01, wcs=wcs, width=256, height=256,
        )
        assert keep[0]       # centre
        assert not keep[1]   # 0.1 deg away in Dec, far outside
        assert not keep[2]   # 5 deg away in RA
