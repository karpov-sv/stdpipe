"""
Image Simulation Examples
==========================

This script demonstrates common use cases for the STDPipe simulation module.
"""

from stdpipe import simulation
import numpy as np


def example_1_basic_simulation():
    """Example 1: Create a basic simulated image with stars and galaxies."""
    print("=" * 60)
    print("Example 1: Basic Image Simulation")
    print("=" * 60)

    result = simulation.simulate_image(
        width=500,
        height=500,
        n_stars=100,
        star_flux_range=(100, 10000),
        star_fwhm=3.0,
        n_galaxies=20,
        galaxy_flux_range=(500, 5000),
        background=1000.0,
        readnoise=10.0,
        verbose=True
    )

    image = result['image']
    catalog = result['catalog']

    print(f"\nCreated image with {len(catalog)} objects")
    print(f"Image statistics: mean={np.mean(image):.1f}, median={np.median(image):.1f}")

    # Show catalog breakdown
    print("\nCatalog breakdown:")
    for obj_type in np.unique(catalog['type']):
        n = np.sum(catalog['type'] == obj_type)
        print(f"  {obj_type}: {n}")

    return result


def example_2_moffat_psf():
    """Example 2: Use Moffat PSF instead of Gaussian."""
    print("\n" + "=" * 60)
    print("Example 2: Moffat PSF Stars")
    print("=" * 60)

    result = simulation.simulate_image(
        width=300,
        height=300,
        n_stars=50,
        star_psf='moffat',           # Use Moffat instead of Gaussian
        star_beta=2.5,               # Moffat beta parameter
        star_fwhm=4.0,
        star_flux_range=(1000, 20000),
        background=1000.0,
        readnoise=10.0,
        verbose=True
    )

    catalog = result['catalog']
    stars = catalog[catalog['type'] == 'star']

    print(f"\nCreated {len(stars)} stars with Moffat PSF")
    print(f"FWHM: {stars['fwhm'][0]:.1f} pixels")
    print(f"PSF type: {stars['psf_type'][0]}")

    return result


def example_3_artifacts():
    """Example 3: Add imaging artifacts for real-bogus training."""
    print("\n" + "=" * 60)
    print("Example 3: Imaging Artifacts")
    print("=" * 60)

    result = simulation.simulate_image(
        width=800,
        height=800,
        n_stars=100,
        n_galaxies=20,
        # Add various artifacts
        n_cosmic_rays=15,
        cosmic_ray_length_range=(10, 60),
        n_hot_pixels=30,
        n_satellites=2,
        satellite_length_range=(200, 500),
        n_bad_columns=1,
        background=1000.0,
        readnoise=10.0,
        verbose=True
    )

    catalog = result['catalog']

    print("\nArtifact summary:")
    real_sources = catalog[catalog['is_real'] == True]
    artifacts = catalog[catalog['is_real'] == False]

    print(f"  Real sources: {len(real_sources)}")
    print(f"  Artifacts: {len(artifacts)}")

    for artifact_type in ['cosmic_ray', 'hot_pixel', 'satellite_trail', 'bad_column']:
        n = np.sum(catalog['type'] == artifact_type)
        if n > 0:
            print(f"    - {artifact_type}: {n}")

    return result


def example_4_diffraction_spikes():
    """Example 4: Add diffraction spikes and optical ghosts to bright stars."""
    print("\n" + "=" * 60)
    print("Example 4: Diffraction Spikes and Optical Ghosts")
    print("=" * 60)

    result = simulation.simulate_image(
        width=500,
        height=500,
        n_stars=30,
        star_flux_range=(1000, 100000),  # Include very bright stars
        star_fwhm=3.5,
        # Enable diffraction spikes and ghosts
        diffraction_spikes=True,
        spike_threshold=20000,           # Add spikes to stars > 20000 ADU
        optical_ghosts=True,
        ghost_threshold=50000,           # Add ghosts to stars > 50000 ADU
        background=1000.0,
        readnoise=10.0,
        verbose=True
    )

    catalog = result['catalog']
    stars = catalog[catalog['type'] == 'star']

    bright_stars = stars[stars['flux'] > 20000]
    very_bright = stars[stars['flux'] > 50000]

    print(f"\nTotal stars: {len(stars)}")
    print(f"Stars with diffraction spikes (flux > 20k): {len(bright_stars)}")
    print(f"Stars with optical ghosts (flux > 50k): {len(very_bright)}")

    return result


def example_5_incremental():
    """Example 5: Build image incrementally with fine control."""
    print("\n" + "=" * 60)
    print("Example 5: Incremental Image Building")
    print("=" * 60)

    # Start with blank image
    image = np.zeros((400, 400))

    # Add faint background stars
    print("Adding faint background stars...")
    cat1 = simulation.add_stars(
        image,
        n=100,
        flux_range=(100, 1000),
        fwhm=2.5,
        psf_type='gaussian'
    )

    # Add bright foreground stars with Moffat PSF
    print("Adding bright foreground stars...")
    cat2 = simulation.add_stars(
        image,
        n=20,
        flux_range=(5000, 20000),
        fwhm=4.0,
        psf_type='moffat',
        beta=2.5
    )

    # Add galaxies
    print("Adding galaxies...")
    cat3 = simulation.add_galaxies(
        image,
        n=15,
        flux_range=(1000, 5000),
        r_eff_range=(5, 15)
    )

    # Add cosmic rays
    print("Adding cosmic rays...")
    cat4 = simulation.add_cosmic_rays(
        image,
        n_rays=10,
        profile='sharp'
    )

    # Add noise
    print("Adding noise...")
    background = 1000.0
    readnoise = 10.0
    image += background
    image += np.random.normal(0, readnoise, image.shape)

    print(f"\nTotal objects added:")
    print(f"  Faint stars: {len(cat1)}")
    print(f"  Bright stars: {len(cat2)}")
    print(f"  Galaxies: {len(cat3)}")
    print(f"  Cosmic rays: {len(cat4)}")

    # Combine catalogs
    from astropy.table import vstack
    combined_catalog = vstack([cat1, cat2, cat3, cat4], join_type='outer')

    result = {
        'image': image,
        'catalog': combined_catalog
    }

    return result


def example_6_galaxy_morphologies():
    """Example 6: Create galaxies with different morphologies."""
    print("\n" + "=" * 60)
    print("Example 6: Galaxy Morphologies")
    print("=" * 60)

    image = np.zeros((600, 600))

    # Spiral galaxies (exponential disks, n=1)
    print("Adding spiral galaxies (n=1.0)...")
    cat_spirals = simulation.add_galaxies(
        image,
        n=10,
        flux_range=(2000, 8000),
        r_eff_range=(10, 20),
        n_range=(0.8, 1.2),          # Sersic n ~ 1
        ellipticity_range=(0.3, 0.7)
    )

    # Elliptical galaxies (de Vaucouleurs, n=4)
    print("Adding elliptical galaxies (n=4.0)...")
    cat_ellipticals = simulation.add_galaxies(
        image,
        n=5,
        flux_range=(3000, 10000),
        r_eff_range=(8, 15),
        n_range=(3.5, 4.5),          # Sersic n ~ 4
        ellipticity_range=(0.2, 0.6)
    )

    # Dwarf galaxies (compact, n=0.5-1)
    print("Adding dwarf galaxies (n=0.5-1.0)...")
    cat_dwarfs = simulation.add_galaxies(
        image,
        n=15,
        flux_range=(500, 2000),
        r_eff_range=(2, 5),
        n_range=(0.5, 1.0)
    )

    print(f"\nGalaxy morphologies:")
    print(f"  Spiral (n~1): {len(cat_spirals)}")
    print(f"  Elliptical (n~4): {len(cat_ellipticals)}")
    print(f"  Dwarf (n~0.5-1): {len(cat_dwarfs)}")

    # Add noise
    image += 1000.0
    image += np.random.normal(0, 10.0, image.shape)

    from astropy.table import vstack
    combined_catalog = vstack([cat_spirals, cat_ellipticals, cat_dwarfs], join_type='outer')

    result = {
        'image': image,
        'catalog': combined_catalog
    }

    return result


def example_7_custom_psf():
    """Example 7: Create custom PSF stamps."""
    print("\n" + "=" * 60)
    print("Example 7: Custom PSF Creation")
    print("=" * 60)

    # Create Gaussian PSF
    psf_gauss = simulation.create_psf_stamp(
        size=25,
        x0=12.0,
        y0=12.0,
        fwhm=3.5,
        psf_type='gaussian',
        pixel_integrated=True
    )

    # Create Moffat PSF
    psf_moffat = simulation.create_psf_stamp(
        size=25,
        x0=12.0,
        y0=12.0,
        fwhm=3.5,
        psf_type='moffat',
        beta=2.5,
        pixel_integrated=True
    )

    print(f"Gaussian PSF:")
    print(f"  Size: {psf_gauss.shape}")
    print(f"  Sum: {np.sum(psf_gauss):.6f}")
    print(f"  Peak: {np.max(psf_gauss):.6f}")

    print(f"\nMoffat PSF:")
    print(f"  Size: {psf_moffat.shape}")
    print(f"  Sum: {np.sum(psf_moffat):.6f}")
    print(f"  Peak: {np.max(psf_moffat):.6f}")

    # Compare profiles
    center = psf_gauss.shape[0] // 2
    profile_gauss = psf_gauss[center, :]
    profile_moffat = psf_moffat[center, :]

    print(f"\nMoffat has broader wings than Gaussian")
    print(f"  Gaussian wing (10 pix from center): {profile_gauss[center+10]:.6f}")
    print(f"  Moffat wing (10 pix from center): {profile_moffat[center+10]:.6f}")
    print(f"  Ratio: {profile_moffat[center+10] / profile_gauss[center+10]:.2f}x")

    return {'psf_gauss': psf_gauss, 'psf_moffat': psf_moffat}


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("STDPipe Image Simulation Examples")
    print("=" * 60 + "\n")

    # Run examples
    result1 = example_1_basic_simulation()
    result2 = example_2_moffat_psf()
    result3 = example_3_artifacts()
    result4 = example_4_diffraction_spikes()
    result5 = example_5_incremental()
    result6 = example_6_galaxy_morphologies()
    result7 = example_7_custom_psf()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

    return {
        'basic': result1,
        'moffat': result2,
        'artifacts': result3,
        'spikes': result4,
        'incremental': result5,
        'galaxies': result6,
        'psf': result7
    }


if __name__ == '__main__':
    results = main()

    # Optionally save one example image
    try:
        from astropy.io import fits
        fits.writeto('simulated_image_example.fits', results['basic']['image'], overwrite=True)
        print("\nSaved example image to 'simulated_image_example.fits'")
    except Exception as e:
        print(f"\nCould not save FITS file: {e}")
