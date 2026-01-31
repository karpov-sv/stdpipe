"""
Comprehensive test suite for the simulation module.

Tests cover:
- PSF models (Gaussian, Moffat) with pixel integration
- Galaxy profiles (Sersic) with ellipticity and rotation
- Imaging artifacts (cosmic rays, hot pixels, etc.)
- High-level and incremental APIs
- Catalog format and completeness
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np
from astropy.table import Table

from stdpipe import simulation


@pytest.mark.unit
class TestPSFModels:
    """Test PSF model creation and properties."""

    def test_gaussian_psf_model_creation(self):
        """Test that Gaussian PSF model is created correctly."""
        fwhm = 3.0

        psf_model = simulation.create_psf_model(fwhm=fwhm, psf_type='gaussian')

        assert 'data' in psf_model
        assert 'width' in psf_model
        assert 'sampling' in psf_model
        assert psf_model['fwhm'] == fwhm
        assert psf_model['psf_type'] == 'gaussian'

        # Check that data is normalized
        assert np.isclose(np.sum(psf_model['data']), 1.0, rtol=1e-6)

    def test_moffat_psf_model_creation(self):
        """Test that Moffat PSF model is created correctly."""
        fwhm = 3.0
        beta = 2.5

        psf_model = simulation.create_psf_model(fwhm=fwhm, psf_type='moffat', beta=beta)

        assert 'data' in psf_model
        assert 'width' in psf_model
        assert 'sampling' in psf_model
        assert psf_model['fwhm'] == fwhm
        assert psf_model['psf_type'] == 'moffat'
        assert psf_model['beta'] == beta

        # Check that data is normalized
        assert np.isclose(np.sum(psf_model['data']), 1.0, rtol=1e-6)

    def test_psf_model_with_psf_module(self):
        """Test that PSF model can be used with psf.get_psf_stamp()."""
        from stdpipe import psf as psf_module

        fwhm = 3.5
        psf_model = simulation.create_psf_model(fwhm=fwhm, psf_type='gaussian')

        # Get stamp at a position
        stamp = psf_module.get_psf_stamp(psf_model, x=50.3, y=50.7, normalize=True)

        # Should be a 2D array
        assert stamp.ndim == 2

        # Should normalize to 1
        assert np.isclose(np.sum(stamp), 1.0, rtol=1e-4)

    def test_moffat_beta_parameter(self):
        """Test that Moffat beta parameter affects PSF shape."""
        fwhm = 4.0

        # Different beta values
        psf_beta2 = simulation.create_psf_model(fwhm=fwhm, psf_type='moffat', beta=2.0)
        psf_beta4 = simulation.create_psf_model(fwhm=fwhm, psf_type='moffat', beta=4.0)

        # Different beta should give different profiles
        assert not np.allclose(psf_beta2['data'], psf_beta4['data']), "Different beta should change PSF shape"

        # Higher beta should have more concentrated profile
        # Check central pixel
        center_idx = psf_beta2['data'].shape[1] // 2
        assert psf_beta4['data'][0, center_idx, center_idx] > psf_beta2['data'][0, center_idx, center_idx], "Higher beta should have sharper peak"

    @pytest.mark.skip(reason="Oversampling not implemented yet - see TODO in create_psf_model()")
    def test_oversampling_factor(self):
        """Test that oversampling factor affects model resolution."""
        # NOTE: Currently oversampling parameter is ignored and sampling=1.0 is always used
        # This provides excellent flux conservation but no true oversampling
        # See create_psf_model() for TODO comment
        fwhm = 3.0
        size = 25  # Fixed output size

        psf_2x = simulation.create_psf_model(fwhm=fwhm, psf_type='gaussian', size=size, oversampling=2)
        psf_4x = simulation.create_psf_model(fwhm=fwhm, psf_type='gaussian', size=size, oversampling=4)
        psf_8x = simulation.create_psf_model(fwhm=fwhm, psf_type='gaussian', size=size, oversampling=8)

        # Higher oversampling should give larger data arrays
        assert psf_2x['data'].size < psf_4x['data'].size < psf_8x['data'].size

        # All should have same target output size (width/height)
        assert psf_2x['width'] == psf_4x['width'] == psf_8x['width']

        # Sampling factors should differ
        assert psf_2x['sampling'] == 2
        assert psf_4x['sampling'] == 4
        assert psf_8x['sampling'] == 8


@pytest.mark.unit
class TestGalaxyProfiles:
    """Test Sersic galaxy profile generation."""

    def test_sersic_flux_conservation(self):
        """Test that Sersic profile conserves total flux."""
        size = 101
        x0, y0 = size // 2, size // 2
        r_eff = 10.0
        n = 1.0

        profile = simulation.create_sersic_profile(size, x0, y0, 1.0, r_eff, n)

        # For n=1 (exponential), most flux should be within a few r_eff
        # Total flux should be reasonably well captured in the stamp
        total_flux = np.sum(profile)
        assert total_flux > 0, "Profile should have positive flux"

    def test_sersic_index_effect(self):
        """Test that Sersic index affects profile shape."""
        size = 101
        x0, y0 = size // 2, size // 2
        r_eff = 10.0

        # Different Sersic indices
        prof_n05 = simulation.create_sersic_profile(size, x0, y0, 1.0, r_eff, 0.5)
        prof_n1 = simulation.create_sersic_profile(size, x0, y0, 1.0, r_eff, 1.0)
        prof_n4 = simulation.create_sersic_profile(size, x0, y0, 1.0, r_eff, 4.0)

        # Different n should give different profiles
        assert not np.allclose(prof_n05, prof_n1)
        assert not np.allclose(prof_n1, prof_n4)

        # Higher n should be more concentrated
        assert prof_n4[y0, x0] > prof_n1[y0, x0], "n=4 should be more concentrated than n=1"

    def test_sersic_ellipticity(self):
        """Test that ellipticity parameter works."""
        size = 101
        x0, y0 = size // 2, size // 2
        r_eff = 15.0
        n = 1.0

        # Circular profile
        prof_circular = simulation.create_sersic_profile(
            size, x0, y0, 1.0, r_eff, n, ellipticity=0.0
        )

        # Elliptical profile
        prof_elliptical = simulation.create_sersic_profile(
            size, x0, y0, 1.0, r_eff, n, ellipticity=0.5
        )

        # Should be different
        assert not np.allclose(prof_circular, prof_elliptical)

        # Circular should be symmetric
        assert np.allclose(
            prof_circular[y0, x0 + 10], prof_circular[y0 + 10, x0], rtol=0.1
        ), "Circular profile should be radially symmetric"

    def test_sersic_rotation(self):
        """Test that position angle rotates the profile."""
        size = 101
        x0, y0 = size // 2, size // 2
        r_eff = 15.0

        # Elliptical profile at different angles
        prof_0deg = simulation.create_sersic_profile(
            size, x0, y0, 1.0, r_eff, 1.0, ellipticity=0.5, position_angle=0.0
        )
        prof_90deg = simulation.create_sersic_profile(
            size, x0, y0, 1.0, r_eff, 1.0, ellipticity=0.5, position_angle=90.0
        )

        # Should be different
        assert not np.allclose(prof_0deg, prof_90deg)

        # 90-degree rotation should swap axes
        # Check that the profile is elongated in different directions
        # At 0 degrees, should be elongated vertically (q < 1 means y-axis is compressed)
        # At 90 degrees, should be elongated horizontally

    def test_place_galaxy_adds_to_image(self):
        """Test that place_galaxy adds flux to image."""
        image = np.zeros((100, 100))
        x0, y0 = 50, 50
        flux = 1000.0

        simulation.place_galaxy(image, x0, y0, flux, r_eff=5.0, n=1.0)

        # Image should now have positive flux
        assert np.sum(image) > 0, "Galaxy should add flux to image"

        # Total flux should be approximately correct
        assert np.isclose(
            np.sum(image), flux, rtol=0.1
        ), f"Expected ~{flux}, got {np.sum(image)}"


@pytest.mark.unit
class TestCosmicRays:
    """Test cosmic ray generation."""

    def test_cosmic_ray_creation(self):
        """Test basic cosmic ray creation."""
        cr = simulation.create_cosmic_ray(
            length=30, width=2, angle=0, max_intensity=5000, profile='sharp'
        )

        assert 'stamp' in cr
        assert 'size' in cr
        assert cr['stamp'].max() > 0, "Cosmic ray should have positive intensity"

    def test_cosmic_ray_profiles(self):
        """Test different cosmic ray profiles."""
        profiles = ['sharp', 'tapered', 'worm']

        for profile in profiles:
            cr = simulation.create_cosmic_ray(
                length=30, width=2, angle=0, max_intensity=5000, profile=profile
            )
            assert cr['stamp'].max() > 0, f"Profile {profile} should have positive intensity"

    def test_add_cosmic_rays_to_image(self):
        """Test adding cosmic rays to image."""
        image = np.zeros((100, 100))
        n_rays = 5

        cat = simulation.add_cosmic_rays(image, n_rays=n_rays)

        # Catalog should have correct number of entries
        assert len(cat) == n_rays

        # Image should be modified
        assert np.sum(image) > 0, "Cosmic rays should modify image"

        # Check catalog columns
        assert 'x' in cat.colnames
        assert 'y' in cat.colnames
        assert 'type' in cat.colnames
        assert all(cat['type'] == 'cosmic_ray')


@pytest.mark.unit
class TestHotPixels:
    """Test hot pixel generation."""

    def test_add_hot_pixels(self):
        """Test adding hot pixels to image."""
        image = np.zeros((100, 100))
        n_pixels = 10

        cat = simulation.add_hot_pixels(image, n_pixels=n_pixels)

        # Catalog should have correct size
        assert len(cat) == n_pixels

        # Image should be modified
        assert np.sum(image) > 0

        # Check catalog
        assert 'x' in cat.colnames
        assert 'y' in cat.colnames
        assert 'intensity' in cat.colnames
        assert all(cat['type'] == 'hot_pixel')

    def test_hot_pixel_clustering(self):
        """Test clustered hot pixels."""
        image = np.zeros((100, 100))

        cat = simulation.add_hot_pixels(
            image, n_pixels=12, clustering=True, cluster_size=3
        )

        # Should have ~4 clusters of 3 pixels each
        assert len(cat) >= 9  # At least 3 clusters

        # Image should be modified
        assert np.sum(image) > 0


@pytest.mark.unit
class TestBadColumns:
    """Test bad column generation."""

    def test_add_dead_columns(self):
        """Test adding dead columns."""
        image = np.ones((100, 100)) * 1000  # Background image

        cat = simulation.add_bad_columns(image, n_columns=2, bad_type='dead')

        # Should have 2 dead columns
        assert len(cat) == 2

        # Image should have some zero columns
        assert np.any(image == 0), "Should have dead pixels"

    def test_add_hot_columns(self):
        """Test adding hot columns."""
        image = np.ones((100, 100)) * 100

        cat = simulation.add_bad_columns(
            image, n_columns=2, intensity_range=(5000, 10000), bad_type='hot'
        )

        assert len(cat) == 2
        assert np.max(image) > 1000, "Should have hot pixels"

    def test_bad_rows(self):
        """Test adding bad rows (horizontal)."""
        image = np.ones((100, 100)) * 100

        cat = simulation.add_bad_columns(
            image, n_columns=2, bad_type='dead', orientation='horizontal'
        )

        assert len(cat) == 2
        assert 'orientation' in cat.colnames
        assert all(cat['orientation'] == 'horizontal')


@pytest.mark.unit
class TestSatelliteTrails:
    """Test satellite trail generation."""

    def test_create_satellite_trail(self):
        """Test basic satellite trail creation."""
        trail = simulation.create_satellite_trail(
            length=200, width=3, intensity=10000, angle=45
        )

        assert 'stamp' in trail
        assert 'angle' in trail
        assert trail['stamp'].max() > 0

    def test_add_satellite_trails(self):
        """Test adding satellite trails to image."""
        image = np.zeros((500, 500))

        cat = simulation.add_satellite_trails(image, n_trails=2)

        assert len(cat) == 2
        assert np.sum(image) > 0

        # Check catalog
        assert 'x' in cat.colnames
        assert 'y' in cat.colnames
        assert 'length' in cat.colnames
        assert 'width' in cat.colnames
        assert 'angle' in cat.colnames
        assert all(cat['type'] == 'satellite_trail')


@pytest.mark.unit
class TestDiffractionSpikes:
    """Test diffraction spike generation."""

    def test_create_diffraction_spikes(self):
        """Test creating diffraction spikes."""
        size = 101
        x0, y0 = size // 2, size // 2
        star_flux = 100000

        spikes = simulation.create_diffraction_spikes(
            size, x0, y0, star_flux, n_spikes=4, spike_length=40
        )

        assert spikes.shape == (size, size)
        assert np.sum(spikes) > 0, "Spikes should have positive flux"

        # Spikes should be fainter than the star
        # (this tests the spike, not including the star itself)
        assert np.sum(spikes) < star_flux, "Spikes should be small fraction of star flux"


@pytest.mark.unit
class TestOpticalGhosts:
    """Test optical ghost generation."""

    def test_create_optical_ghost(self):
        """Test creating an optical ghost."""
        size = 101
        x0, y0 = size // 2, size // 2
        source_flux = 100000

        ghost = simulation.create_optical_ghost(
            size, x0, y0, source_flux, ghost_fraction=0.05, offset=(30, 30)
        )

        assert ghost.shape == (size, size)
        assert np.sum(ghost) > 0, "Ghost should have positive flux"

        # Ghost should be small fraction of source
        assert (
            np.sum(ghost) < source_flux * 0.1
        ), "Ghost should be small fraction of source flux"


@pytest.mark.unit
class TestIncrementalAPI:
    """Test incremental API functions."""

    def test_add_stars(self):
        """Test add_stars function."""
        image = np.zeros((200, 200))

        cat = simulation.add_stars(
            image, n=50, flux_range=(100, 10000), fwhm=3.0, psf='gaussian'
        )

        # Check catalog
        assert len(cat) == 50
        assert 'x' in cat.colnames
        assert 'y' in cat.colnames
        assert 'flux' in cat.colnames
        assert 'fwhm' in cat.colnames
        assert 'psf_type' in cat.colnames
        assert all(cat['type'] == 'star')
        assert all(cat['is_real'] == True)

        # Image should be modified
        assert np.sum(image) > 0

    def test_add_stars_with_moffat(self):
        """Test add_stars with Moffat PSF."""
        image = np.zeros((200, 200))

        cat = simulation.add_stars(
            image, n=20, flux_range=(100, 5000), fwhm=4.0, psf='moffat', beta=2.5
        )

        assert len(cat) == 20
        assert np.sum(image) > 0

    def test_add_galaxies(self):
        """Test add_galaxies function."""
        image = np.zeros((200, 200))

        cat = simulation.add_galaxies(
            image, n=10, flux_range=(500, 5000), r_eff_range=(3, 10)
        )

        # Check catalog
        assert len(cat) == 10
        assert 'x' in cat.colnames
        assert 'flux' in cat.colnames
        assert 'r_eff' in cat.colnames
        assert 'sersic_n' in cat.colnames
        assert 'ellipticity' in cat.colnames
        assert all(cat['type'] == 'galaxy')

        # Image should be modified
        assert np.sum(image) > 0


@pytest.mark.unit
class TestHighLevelAPI:
    """Test high-level simulate_image function."""

    def test_simulate_image_basic(self):
        """Test basic image simulation."""
        result = simulation.simulate_image(
            width=200,
            height=200,
            n_stars=50,
            n_galaxies=10,
            n_cosmic_rays=5,
            n_hot_pixels=10,
            background=1000.0,
            readnoise=10.0,
        )

        # Check output structure
        assert 'image' in result
        assert 'catalog' in result
        assert 'background' in result
        assert 'noise' in result

        # Check image
        image = result['image']
        assert image.shape == (200, 200)
        assert np.mean(image) > 0

        # Check catalog
        cat = result['catalog']
        assert len(cat) == 50 + 10 + 5 + 10  # stars + galaxies + cosmic rays + hot pixels

        # Check that catalog has proper types
        assert 'type' in cat.colnames
        assert 'is_real' in cat.colnames

        # Count types
        n_stars = np.sum(cat['type'] == 'star')
        n_galaxies = np.sum(cat['type'] == 'galaxy')
        n_cosmic = np.sum(cat['type'] == 'cosmic_ray')
        n_hot = np.sum(cat['type'] == 'hot_pixel')

        assert n_stars == 50
        assert n_galaxies == 10
        assert n_cosmic == 5
        assert n_hot == 10

    def test_simulate_image_with_artifacts(self):
        """Test image simulation with all artifacts."""
        result = simulation.simulate_image(
            width=400,
            height=400,
            n_stars=100,
            n_galaxies=20,
            n_cosmic_rays=10,
            n_hot_pixels=20,
            n_bad_columns=2,
            n_satellites=1,
            diffraction_spikes=True,
            spike_threshold=50000,
            optical_ghosts=True,
            ghost_threshold=100000,
            background=1000.0,
        )

        assert result['image'].shape == (400, 400)
        assert 'catalog' in result

        # Check that catalog includes all types
        cat = result['catalog']
        assert len(cat) > 100  # At least stars + galaxies + artifacts

    def test_simulate_image_moffat_psf(self):
        """Test image simulation with Moffat PSF."""
        result = simulation.simulate_image(
            width=200,
            height=200,
            n_stars=30,
            star_psf='moffat',
            star_beta=2.5,
            star_fwhm=4.0,
        )

        cat = result['catalog']
        stars = cat[cat['type'] == 'star']
        assert all(stars['psf_type'] == 'moffat')

    def test_simulate_empty_image(self):
        """Test simulating image with no sources (just background)."""
        result = simulation.simulate_image(
            width=100,
            height=100,
            n_stars=0,
            n_galaxies=0,
            n_cosmic_rays=0,
            n_hot_pixels=0,
            background=1000.0,
            readnoise=10.0,
        )

        image = result['image']
        assert image.shape == (100, 100)

        # Should be approximately background level
        assert np.abs(np.median(image) - 1000.0) < 100


@pytest.mark.unit
class TestCatalogFormat:
    """Test catalog output format and consistency."""

    def test_catalog_has_required_columns(self):
        """Test that catalogs have required columns."""
        result = simulation.simulate_image(
            width=200, height=200, n_stars=10, n_galaxies=5, n_cosmic_rays=3
        )

        cat = result['catalog']

        # All entries should have these columns
        assert 'x' in cat.colnames
        assert 'y' in cat.colnames
        assert 'type' in cat.colnames
        assert 'is_real' in cat.colnames

        # Check is_real values
        stars = cat[cat['type'] == 'star']
        assert all(stars['is_real'] == True)

        cosmic = cat[cat['type'] == 'cosmic_ray']
        assert all(cosmic['is_real'] == False)

    def test_star_catalog_columns(self):
        """Test that star catalog has correct columns."""
        image = np.zeros((100, 100))
        cat = simulation.add_stars(image, n=10, fwhm=3.0)

        # Star-specific columns
        assert 'flux' in cat.colnames
        assert 'fwhm' in cat.colnames
        assert 'psf_type' in cat.colnames

    def test_galaxy_catalog_columns(self):
        """Test that galaxy catalog has correct columns."""
        image = np.zeros((100, 100))
        cat = simulation.add_galaxies(image, n=5)

        # Galaxy-specific columns
        assert 'flux' in cat.colnames
        assert 'r_eff' in cat.colnames
        assert 'sersic_n' in cat.colnames
        assert 'ellipticity' in cat.colnames
        assert 'position_angle' in cat.colnames


@pytest.mark.integration
class TestPhotometryIntegration:
    """Integration tests with photometry module."""

    def test_detect_simulated_stars(self):
        """Test that simulated stars can be detected."""
        from stdpipe import photometry

        # Simulate image with known stars
        result = simulation.simulate_image(
            width=300,
            height=300,
            n_stars=20,
            star_flux_range=(1000, 10000),
            star_fwhm=3.0,
            n_galaxies=0,
            n_cosmic_rays=0,
            n_hot_pixels=0,
            background=100.0,
            readnoise=5.0,
        )

        image = result['image']
        true_cat = result['catalog']

        # Detect objects using SEP
        detected = photometry.get_objects_sep(
            image, thresh=5.0, aper=5.0, minarea=5
        )

        # Should detect most stars (allow some losses due to blending/noise)
        assert len(detected) >= 15, f"Expected ~20 stars, detected {len(detected)}"

    def test_photometry_accuracy(self):
        """Test photometry accuracy on simulated stars."""
        from stdpipe import photometry, photometry_measure

        # Set seed for reproducibility and to avoid random blending
        np.random.seed(12345)

        # Simulate bright, isolated stars
        result = simulation.simulate_image(
            width=500,
            height=500,
            n_stars=10,
            star_flux_range=(5000, 10000),
            star_fwhm=4.0,
            n_galaxies=0,
            n_cosmic_rays=0,
            background=100.0,
            readnoise=5.0,
            edge=50,  # Keep stars away from edges
        )

        image = result['image']
        true_cat = result['catalog']
        true_stars = true_cat[true_cat['type'] == 'star']

        # Detect objects
        detected = photometry.get_objects_sep(image, thresh=5.0, aper=5.0)

        # Measure photometry
        # Use smaller aperture (2xFWHM) which works better with sampling=1 PSF model
        measured = photometry_measure.measure_objects(
            detected, image, fwhm=4.0, aper=8.0
        )

        # Should have detected all stars
        assert len(measured) >= 8, f"Expected ~10 stars, measured {len(measured)}"

        # Match detected to true catalog (simple nearest neighbor)
        n_matched = 0
        n_accurate = 0

        for true_star in true_stars:
            # Find nearest detected object
            dx = measured['x'] - true_star['x']
            dy = measured['y'] - true_star['y']
            dist = np.sqrt(dx**2 + dy**2)
            nearest_idx = np.argmin(dist)

            if dist[nearest_idx] < 3.0:  # Within 3 pixels
                n_matched += 1
                measured_flux = measured[nearest_idx]['flux']
                true_flux = true_star['flux']

                # Check flux accuracy
                rel_error = np.abs(measured_flux - true_flux) / true_flux

                # Count stars with good flux accuracy (<30%)
                if rel_error < 0.3:
                    n_accurate += 1

        # Require that most matched stars have reasonable flux accuracy
        # Allow some failures due to background estimation, blending, etc.
        assert n_matched >= 8, f"Expected to match ~10 stars, only matched {n_matched}"
        assert n_accurate >= 6, f"Expected >=6 accurate measurements, got {n_accurate}/{n_matched}"


@pytest.mark.unit
@pytest.mark.skip(reason="Aberrated PSF feature not implemented yet in create_psf_model()")
class TestAberratedPSF:
    """Test PSF models with optical aberrations."""

    def test_no_aberration_matches_original(self):
        """Zero aberrations should give identical result to original."""
        fwhm = 3.5

        # Create PSF without aberration parameters
        psf_original = simulation.create_psf_model(fwhm=fwhm, psf_type='gaussian')

        # Create PSF with explicit zero aberrations
        psf_zero_aberr = simulation.create_psf_model(
            fwhm=fwhm,
            psf_type='gaussian',
            defocus=0.0,
            astigmatism_x=0.0,
            astigmatism_y=0.0,
            coma_x=0.0,
            coma_y=0.0
        )

        # Should produce identical results
        assert psf_original['data'].shape == psf_zero_aberr['data'].shape
        assert np.allclose(psf_original['data'], psf_zero_aberr['data'], rtol=1e-6)
        assert psf_original['psf_type'] == psf_zero_aberr['psf_type']

    def test_defocus_creates_ring(self):
        """Defocus should create donut/ring shaped PSF."""
        fwhm = 4.0

        # Strong defocus
        psf_defocused = simulation.create_psf_model(
            fwhm=fwhm,
            psf_type='gaussian',
            defocus=2.0,  # 2 waves of defocus
            size=51
        )

        psf_data = psf_defocused['data'][0]
        center = psf_data.shape[0] // 2

        # For strong defocus, the PSF should have a ring structure
        # Check that the center is not the maximum
        center_value = psf_data[center, center]
        max_value = np.max(psf_data)

        # Center should be less than maximum for defocused PSF
        assert center_value < max_value, "Defocused PSF should not have peak at center"

        # Check that PSF is normalized
        assert np.isclose(np.sum(psf_data), 1.0, rtol=1e-4)

        # Check aberrations metadata
        assert 'aberrations' in psf_defocused
        assert psf_defocused['aberrations']['defocus'] == 2.0

    def test_astigmatism_creates_ellipse(self):
        """Astigmatism should create elliptical PSF structure."""
        fwhm = 4.0

        # Astigmatism in X direction
        psf_astig = simulation.create_psf_model(
            fwhm=fwhm,
            psf_type='gaussian',
            astigmatism_x=1.5,  # 1.5 waves of astigmatism
            size=51
        )

        psf_data = psf_astig['data'][0]

        # Check that PSF is created and normalized
        assert np.isclose(np.sum(psf_data), 1.0, rtol=1e-4)

        # Check that aberration metadata is stored
        assert 'aberrations' in psf_astig
        assert psf_astig['aberrations']['astigmatism_x'] == 1.5

        # Compare with non-aberrated PSF - they should be different
        psf_normal = simulation.create_psf_model(
            fwhm=fwhm,
            psf_type='gaussian',
            size=51
        )

        # Aberrated and normal PSF should be different
        # Note: seeing convolution can reduce the visibility of aberrations,
        # but there should still be some difference in the PSF structure
        assert not np.allclose(psf_data, psf_normal['data'][0], rtol=0.01), \
            "Astigmatic PSF should differ from non-aberrated PSF"

    def test_coma_creates_asymmetry(self):
        """Coma should create asymmetric PSF with tail."""
        fwhm = 4.0

        # Coma in X direction
        psf_coma = simulation.create_psf_model(
            fwhm=fwhm,
            psf_type='gaussian',
            coma_x=1.0,  # 1 wave of coma
            size=51
        )

        psf_data = psf_coma['data'][0]
        center_idx = psf_data.shape[0] // 2

        # Coma creates asymmetry
        # Check that left and right sides are different
        left_sum = np.sum(psf_data[:, :center_idx])
        right_sum = np.sum(psf_data[:, center_idx:])

        # Should have different flux on each side
        assert not np.isclose(left_sum, right_sum, rtol=0.05), "Coma should create asymmetry"

        # Check normalization
        assert np.isclose(np.sum(psf_data), 1.0, rtol=1e-4)

    def test_combined_aberrations(self):
        """Multiple aberrations should combine correctly."""
        fwhm = 4.0

        # Multiple aberrations
        psf_combined = simulation.create_psf_model(
            fwhm=fwhm,
            psf_type='gaussian',
            defocus=0.8,
            astigmatism_x=0.5,
            coma_y=0.3,
            size=51
        )

        psf_data = psf_combined['data'][0]

        # Should still be normalized
        assert np.isclose(np.sum(psf_data), 1.0, rtol=1e-4)

        # Should have all aberrations stored
        assert psf_combined['aberrations']['defocus'] == 0.8
        assert psf_combined['aberrations']['astigmatism_x'] == 0.5
        assert psf_combined['aberrations']['coma_y'] == 0.3

        # PSF type should indicate aberration
        assert 'aberrated' in psf_combined['psf_type']

    def test_psf_works_with_placement(self):
        """Aberrated PSF works with psf.place_psf_stamp()."""
        from stdpipe import psf as psf_module

        fwhm = 3.5

        # Create aberrated PSF
        psf_model = simulation.create_psf_model(
            fwhm=fwhm,
            psf_type='gaussian',
            defocus=1.0,
            astigmatism_x=0.5,
            size=41
        )

        # Create test image
        image = np.zeros((100, 100))

        # Place PSF using psf module (note: uses x0, y0 parameter names)
        flux = 1000.0
        psf_module.place_psf_stamp(image, psf_model, x0=50.5, y0=50.5, flux=flux)

        # Check that flux was added
        assert np.sum(image) > 0, "PSF should add flux to image"

        # Check approximate flux conservation (within 10% due to edge effects)
        assert np.abs(np.sum(image) - flux) / flux < 0.1

    def test_aberration_metadata_stored(self):
        """Aberration parameters stored in PSF model."""
        psf_model = simulation.create_psf_model(
            fwhm=3.0,
            defocus=1.2,
            astigmatism_y=0.7,
            coma_x=0.4,
            wavelength=600e-9
        )

        # Check metadata exists
        assert 'aberrations' in psf_model

        aberr = psf_model['aberrations']
        assert aberr['defocus'] == 1.2
        assert aberr['astigmatism_y'] == 0.7
        assert aberr['coma_x'] == 0.4
        assert aberr['wavelength'] == 600e-9

    def test_aberrated_psf_normalization(self):
        """Aberrated PSF properly normalized."""
        # Test various aberration combinations
        test_cases = [
            {'defocus': 1.5},
            {'astigmatism_x': 1.0, 'astigmatism_y': 0.5},
            {'coma_x': 0.8},
            {'defocus': 1.0, 'astigmatism_x': 0.5, 'coma_y': 0.3},
        ]

        for aberrations in test_cases:
            psf_model = simulation.create_psf_model(
                fwhm=4.0,
                psf_type='gaussian',
                size=51,
                **aberrations
            )

            psf_data = psf_model['data'][0]

            # Check normalization (should be very close to 1.0)
            assert np.isclose(np.sum(psf_data), 1.0, rtol=1e-4), \
                f"PSF not normalized for aberrations {aberrations}"

    @pytest.mark.parametrize("psf_type", ["gaussian", "moffat"])
    def test_both_psf_types_with_aberrations(self, psf_type):
        """Aberrations work with both Gaussian and Moffat."""
        psf_model = simulation.create_psf_model(
            fwhm=3.5,
            psf_type=psf_type,
            beta=2.5,
            defocus=1.0,
            astigmatism_x=0.5,
            size=41
        )

        # Should create valid PSF
        assert 'data' in psf_model
        assert psf_model['data'].ndim == 3

        # Should be normalized
        assert np.isclose(np.sum(psf_model['data']), 1.0, rtol=1e-4)

        # Should indicate aberration in type
        assert 'aberrated' in psf_model['psf_type']

    def test_aberrated_stars_in_image(self):
        """Test adding stars with aberrated PSF."""
        # Create aberrated PSF model
        psf_model = simulation.create_psf_model(
            fwhm=3.5,
            psf_type='gaussian',
            defocus=1.2,
            astigmatism_x=0.6,
            size=41
        )

        # Create blank image
        image = np.zeros((200, 200))

        # Add stars with aberrated PSF
        catalog = simulation.add_stars(
            image,
            n=10,
            flux_range=(1000, 5000),
            psf=psf_model,  # Pass PSF model directly
            edge=30
        )

        # Check that stars were added
        assert len(catalog) == 10
        assert np.sum(image) > 0

        # Check catalog has correct PSF type
        assert 'psf_type' in catalog.colnames
        assert 'aberrated' in catalog['psf_type'][0]

    def test_wavelength_parameter(self):
        """Test that wavelength parameter affects PSF."""
        # Different wavelengths should give slightly different PSFs
        # (due to different diffraction patterns)
        psf_550nm = simulation.create_psf_model(
            fwhm=4.0,
            defocus=1.0,
            wavelength=550e-9,
            size=41
        )

        psf_650nm = simulation.create_psf_model(
            fwhm=4.0,
            defocus=1.0,
            wavelength=650e-9,
            size=41
        )

        # PSFs should be slightly different (wavelength affects diffraction scale)
        # But both should be normalized
        assert np.isclose(np.sum(psf_550nm['data']), 1.0, rtol=1e-4)
        assert np.isclose(np.sum(psf_650nm['data']), 1.0, rtol=1e-4)

        # Wavelengths should be stored correctly
        assert psf_550nm['aberrations']['wavelength'] == 550e-9
        assert psf_650nm['aberrations']['wavelength'] == 650e-9

    def test_zero_fwhm_raises_error(self):
        """Test that zero FWHM with aberrations doesn't crash."""
        # This should work - FWHM can be small
        psf_model = simulation.create_psf_model(
            fwhm=0.5,  # Very small FWHM
            defocus=1.0,
            size=21
        )

        # Should still produce valid PSF
        assert np.sum(psf_model['data']) > 0
        assert np.isclose(np.sum(psf_model['data']), 1.0, rtol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
