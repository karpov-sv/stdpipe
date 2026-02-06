"""
Image simulation module for generating realistic stellar fields with contaminants.

This module provides functions for creating simulated astronomical images with:
- Stars with Gaussian and Moffat PSF (pixel-integrated)
- Extended sources (galaxies with Sersic profiles)
- Imaging artifacts (cosmic rays, hot pixels, bad columns, satellite trails)
- Diffraction spikes and optical ghosts for bright sources

Designed for real-bogus classifier development and photometry testing.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.table import Table
from scipy.ndimage import rotate, gaussian_filter


def create_psf_model(
    fwhm=3.0, psf_type='gaussian', beta=2.5, size=None, oversampling=2,
    defocus=0.0, astigmatism_x=0.0, astigmatism_y=0.0,
    coma_x=0.0, coma_y=0.0, wavelength=550e-9, pupil_diameter=1.0,
):
    """
    Create an oversampled PSF model compatible with PSFEx format.

    This creates a PSF model that can be used with psf.place_psf_stamp() and other
    STDPipe routines. Uses oversampling for accurate flux conservation and fast
    evaluation with sub-pixel positioning.

    Optical aberrations can be added via Zernike polynomial coefficients (Noll ordering).
    When any aberration coefficient is non-zero, a diffraction PSF is computed via
    Fourier optics and convolved with the atmospheric seeing PSF (Gaussian or Moffat).
    When all coefficients are zero, the existing analytical path is used unchanged.

    :param fwhm: Full width at half maximum in pixels
    :param psf_type: Type of PSF model ('gaussian' or 'moffat')
    :param beta: Moffat beta parameter (default 2.5, only used for psf_type='moffat')
    :param size: Output stamp size in pixels (after downsampling). If None, automatically sized to capture ~99% of PSF flux (≥8*FWHM)
    :param oversampling: Oversampling factor (default 2)
    :param defocus: Zernike Z4 defocus coefficient in waves (default 0.0)
    :param astigmatism_x: Zernike Z5 oblique astigmatism coefficient in waves (default 0.0)
    :param astigmatism_y: Zernike Z6 vertical astigmatism coefficient in waves (default 0.0)
    :param coma_x: Zernike Z7 vertical coma coefficient in waves (default 0.0)
    :param coma_y: Zernike Z8 horizontal coma coefficient in waves (default 0.0)
    :param wavelength: Observation wavelength in meters (default 550e-9, affects diffraction scale)
    :param pupil_diameter: Pupil diameter in meters (default 1.0, affects diffraction scale)
    :returns: Dictionary with PSFEx-compatible structure containing the PSF model

    """

    # Auto-size to capture full PSF if not specified
    # For Gaussian/Moffat PSF, significant flux extends to ~4 FWHM from center
    # Use 8*FWHM total size to capture ~99% of flux
    if size is None:
        size = int(np.ceil(fwhm * 8))
        if size % 2 == 0:
            size += 1

    # Oversampled data grid following PSFEx convention:
    # - data array has size * oversampling pixels per side
    # - sampling = 1/oversampling (image pixels per data pixel; <1 means oversampled)
    # - get_psf_stamp() output size: floor(width * sampling / 2) * 2 + 1 = size (constant)
    psf_data_size = size * oversampling
    psf_sampling = 1.0 / oversampling  # PSFEx convention: image pixels per data pixel
    psf_width = float(psf_data_size)   # Data array dimension
    psf_height = float(psf_data_size)

    # Center of the oversampled grid
    center = (psf_data_size - 1) / 2.0

    # Create coordinate grid in data-pixel space
    y, x = np.mgrid[0:psf_data_size, 0:psf_data_size]

    if psf_type.lower() == 'gaussian':
        # Pixel-integrated Gaussian PSF using error function
        # This properly accounts for flux integrated over each data pixel area
        # Data pixel [i-0.5, i+0.5] covers image space [(i-0.5)*sampling, (i+0.5)*sampling]
        from scipy.special import erf

        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # in image pixels
        sqrt2_sigma = np.sqrt(2) * sigma

        # X-direction: integrate Gaussian over each data pixel, converting to image space
        erf_xp = erf((x + 0.5 - center) * psf_sampling / sqrt2_sigma)
        erf_xm = erf((x - 0.5 - center) * psf_sampling / sqrt2_sigma)
        integral_x = 0.5 * (erf_xp - erf_xm)

        # Y-direction integration
        erf_yp = erf((y + 0.5 - center) * psf_sampling / sqrt2_sigma)
        erf_ym = erf((y - 0.5 - center) * psf_sampling / sqrt2_sigma)
        integral_y = 0.5 * (erf_yp - erf_ym)

        # Combined 2D pixel-integrated PSF
        psf_data = integral_x * integral_y

    elif psf_type.lower() == 'moffat':
        # Moffat PSF with pixel integration
        # For Moffat, analytical pixel integration is complex, so we use
        # supersampling for accurate pixel integration

        alpha = fwhm / (2 * np.sqrt(2 ** (1.0 / beta) - 1))

        # Use 5x supersampling for pixel integration (vectorized)
        # Offsets are in data-pixel space; convert to image space with psf_sampling
        supersample = 5
        offsets = np.linspace(-0.5 + 0.5 / supersample, 0.5 - 0.5 / supersample, supersample)

        psf_data = np.zeros((psf_data_size, psf_data_size))
        for dy in offsets:
            for dx in offsets:
                r = np.sqrt(((x + dx - center) * psf_sampling) ** 2 + ((y + dy - center) * psf_sampling) ** 2)
                psf_data += 1.0 / (1 + (r / alpha) ** 2) ** beta
        psf_data /= supersample ** 2

    else:
        raise ValueError(
            f"Unknown psf_type '{psf_type}'. Supported types: 'gaussian', 'moffat'"
        )

    # Normalize
    psf_data /= np.sum(psf_data)

    # Apply optical aberrations if any are non-zero
    has_aberrations = any([defocus, astigmatism_x, astigmatism_y, coma_x, coma_y])

    if has_aberrations:
        from scipy.signal import fftconvolve

        # Build pupil grid in polar coordinates on the unit disk
        N = psf_data_size
        coords = np.linspace(-1, 1, N)
        xx, yy = np.meshgrid(coords, coords)
        rho = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
        aperture = (rho <= 1.0).astype(float)

        # Wavefront from Noll-ordered Zernike polynomials Z4-Z8
        W = np.zeros_like(rho)
        if defocus:
            # Z4: sqrt(3) * (2*rho^2 - 1)
            W += defocus * np.sqrt(3) * (2 * rho**2 - 1)
        if astigmatism_x:
            # Z5: sqrt(6) * rho^2 * sin(2*theta)
            W += astigmatism_x * np.sqrt(6) * rho**2 * np.sin(2 * theta)
        if astigmatism_y:
            # Z6: sqrt(6) * rho^2 * cos(2*theta)
            W += astigmatism_y * np.sqrt(6) * rho**2 * np.cos(2 * theta)
        if coma_x:
            # Z7: sqrt(8) * (3*rho^3 - 2*rho) * sin(theta)
            W += coma_x * np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.sin(theta)
        if coma_y:
            # Z8: sqrt(8) * (3*rho^3 - 2*rho) * cos(theta)
            W += coma_y * np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.cos(theta)

        # Scale wavefront by wavelength ratio to make PSF wavelength-dependent
        # Reference wavelength is 550nm; different wavelengths scale the
        # effective aberration strength (shorter wavelength = stronger effect)
        wavelength_ref = 550e-9
        W *= wavelength_ref / wavelength

        # Complex pupil function
        pupil = aperture * np.exp(2j * np.pi * W)

        # Diffraction PSF via FFT
        E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupil)))
        psf_diffraction = np.abs(E)**2
        psf_diffraction /= np.sum(psf_diffraction)

        # Convolve diffraction PSF with seeing PSF (already computed as psf_data)
        psf_data = fftconvolve(psf_diffraction, psf_data, mode='same')
        psf_data /= np.sum(psf_data)

        psf_type_name = psf_type.lower() + '_aberrated'
    else:
        psf_type_name = psf_type.lower()

    # Create PSFEx-compatible structure
    # Add polynomial dimension (ncoeffs=1 for constant PSF)
    psf_data_3d = psf_data[np.newaxis, :, :]

    psf_model = {
        'data': psf_data_3d,
        'width': psf_width,
        'height': psf_height,
        'sampling': psf_sampling,
        'fwhm': fwhm,
        'degree': 0,  # Constant PSF (no position dependence)
        'ncoeffs': 1,
        'x0': 0.0,  # Reference position
        'y0': 0.0,
        'sx': 1.0,  # Scaling factors
        'sy': 1.0,
        'psf_type': psf_type_name,
        'beta': beta if psf_type.lower() == 'moffat' else None,
    }

    if has_aberrations:
        psf_model['aberrations'] = {
            'defocus': defocus,
            'astigmatism_x': astigmatism_x,
            'astigmatism_y': astigmatism_y,
            'coma_x': coma_x,
            'coma_y': coma_y,
            'wavelength': wavelength,
            'pupil_diameter': pupil_diameter,
        }

    return psf_model


def create_sersic_profile(
    size, x0, y0, amplitude=1.0, r_eff=5.0, n=1.0, ellipticity=0.0, position_angle=0.0
):
    """
    Create a Sersic profile for galaxy simulation.

    The Sersic profile is: I(r) = amplitude * exp(-b_n * ((r/r_eff)^(1/n) - 1))
    where b_n ≈ 2n - 0.324 for n >= 0.5 (more accurate formula used internally)

    Common cases:
    - n=0.5: Gaussian-like profile
    - n=1.0: Exponential disk (typical spiral galaxy disk)
    - n=4.0: de Vaucouleurs profile (typical elliptical galaxy)

    :param size: Size of the profile stamp (must be odd)
    :param x0: X center position within the stamp
    :param y0: Y center position within the stamp
    :param amplitude: Central surface brightness amplitude
    :param r_eff: Effective radius in pixels (half-light radius)
    :param n: Sersic index (shape parameter)
    :param ellipticity: Ellipticity (0 = circular, 0.5 = moderate, 0.9 = very elongated)
    :param position_angle: Position angle in degrees (0 = vertical, increases CCW)
    :returns: 2D numpy array of shape (size, size) with Sersic profile

    """

    # Compute b_n parameter using more accurate formula
    # For n >= 0.5: b_n ≈ 2n - 1/3 + 4/(405n) + 46/(25515n^2) + ...
    # For simplicity, use: b_n ≈ 1.9992n - 0.3271 for n > 0.5
    if n > 0.5:
        b_n = 1.9992 * n - 0.3271
    else:
        b_n = 0.01  # Very broad profile

    # Create coordinate grid
    y, x = np.mgrid[0:size, 0:size]
    x = x - x0
    y = y - y0

    # Apply rotation
    if position_angle != 0:
        theta = np.radians(position_angle)
        x_rot = x * np.cos(theta) + y * np.sin(theta)
        y_rot = -x * np.sin(theta) + y * np.cos(theta)
        x, y = x_rot, y_rot

    # Apply ellipticity: convert ellipticity to axis ratio
    # ellipticity = 1 - b/a, so b/a = 1 - ellipticity
    q = 1 - ellipticity  # Axis ratio (minor/major)
    if q <= 0:
        q = 0.01  # Avoid division by zero

    # Elliptical radius
    r = np.sqrt(x**2 + (y / q) ** 2)

    # Sersic profile
    profile = amplitude * np.exp(-b_n * ((r / r_eff) ** (1.0 / n) - 1))

    return profile


def place_galaxy(
    image, x0, y0, flux, r_eff=5.0, n=1.0, ellipticity=0.0, position_angle=0.0
):
    """
    Place a galaxy with Sersic profile into an image.

    The galaxy stamp is created, scaled to the requested flux, and added to the image.
    Similar to psf.place_psf_stamp() but for extended sources.

    The image is modified in-place.

    :param image: Target image (2D numpy array)
    :param x0: X coordinate where to place the galaxy center
    :param y0: Y coordinate where to place the galaxy center
    :param flux: Total integrated flux in ADU units
    :param r_eff: Effective radius in pixels
    :param n: Sersic index
    :param ellipticity: Ellipticity (0 = circular, 1 = line)
    :param position_angle: Position angle in degrees

    """

    # Determine stamp size (should contain most of the flux)
    # For Sersic profiles, most flux is within ~5-8 * r_eff
    size = int(np.ceil(r_eff * 8)) * 2 + 1

    # Integer coordinates and sub-pixel offset
    ix, iy = int(np.round(x0)), int(np.round(y0))
    dx_sub = x0 - ix
    dy_sub = y0 - iy

    # Create galaxy stamp centered with sub-pixel offset
    stamp_x0 = size // 2 + dx_sub
    stamp_y0 = size // 2 + dy_sub

    stamp = create_sersic_profile(
        size, stamp_x0, stamp_y0, 1.0, r_eff, n, ellipticity, position_angle
    )

    # Normalize and scale to requested flux
    stamp_sum = np.sum(stamp)
    if stamp_sum > 0:
        stamp = stamp / stamp_sum * flux
    else:
        return  # Degenerate profile

    # Integer coordinates inside the stamp
    y, x = np.mgrid[0 : stamp.shape[0], 0 : stamp.shape[1]]

    # Corresponding image pixels
    y1, x1 = np.mgrid[0 : stamp.shape[0], 0 : stamp.shape[1]]
    x1 += int(np.round(x0) - np.floor(stamp.shape[1] / 2))
    y1 += int(np.round(y0) - np.floor(stamp.shape[0] / 2))

    # Crop coordinates outside target image
    idx = np.isfinite(stamp)
    idx &= (x1 >= 0) & (x1 < image.shape[1])
    idx &= (y1 >= 0) & (y1 < image.shape[0])

    # Add the stamp to the image
    image[y1[idx], x1[idx]] += stamp[y[idx], x[idx]]


def create_cosmic_ray(length, width, angle, max_intensity, profile='sharp'):
    """
    Create a cosmic ray track.

    Cosmic rays appear as elongated tracks with sharp edges, oriented randomly.

    :param length: Length of the track in pixels
    :param width: Width of the track in pixels
    :param angle: Angle in degrees (0 = horizontal, increases CCW)
    :param max_intensity: Peak intensity in ADU
    :param profile: Intensity profile ('sharp', 'tapered', 'worm')
    :returns: Dictionary with 'stamp' (2D array), 'size' (stamp dimensions)

    """

    # Create a horizontal track on an oversized canvas
    canvas_size = int(np.ceil(max(length, width) * 3))
    canvas = np.zeros((canvas_size, canvas_size))

    # Center of canvas
    cx, cy = canvas_size // 2, canvas_size // 2

    # Create horizontal track
    track_length = int(np.ceil(length))
    track_width = int(np.ceil(width))

    if profile == 'sharp':
        # Sharp-edged rectangular track
        y_start = cy - track_width // 2
        y_end = y_start + track_width
        x_start = cx - track_length // 2
        x_end = x_start + track_length
        canvas[y_start:y_end, x_start:x_end] = max_intensity

    elif profile == 'tapered':
        # Tapered track (intensity decreases toward ends, symmetric)
        y_start = cy - track_width // 2
        y_end = y_start + track_width
        x_start = cx - track_length // 2
        x_end = x_start + track_length
        # Symmetric taper: ramps from 0.2 at edges to 1.0 at center
        taper = np.linspace(-1.0, 1.0, track_length)
        taper = 0.2 + 0.8 * (1.0 - np.abs(taper))
        canvas[y_start:y_end, x_start:x_end] = max_intensity * taper[None, :]

    elif profile == 'worm':
        # Worm-like track with slight curvature
        x_coords = np.arange(track_length) - track_length // 2 + cx
        # Add sinusoidal variation
        y_variation = np.sin(np.linspace(0, 2 * np.pi, track_length)) * (track_width * 1)
        y_coords = y_variation + cy

        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ix, iy = int(np.round(x)), int(np.round(y))
            # Draw small blob at each position
            for dy in range(-track_width // 2, -track_width // 2 + track_width):
                for dx in range(-1, 2):
                    if i + dx < 0 or i + dx >= track_length:
                        continue
                    py, px = iy + dy, ix + dx
                    if 0 <= py < canvas_size and 0 <= px < canvas_size:
                        canvas[py, px] = max_intensity

    else:
        raise ValueError(f"Unknown profile '{profile}'. Use 'sharp', 'tapered', or 'worm'")

    # Rotate the canvas
    if angle != 0:
        # Cosmics are supposed to be sharp so let's rotate without interpolation
        canvas = rotate(canvas, angle, reshape=False, order=0)

    # Crop to minimal bounding box
    nonzero = np.nonzero(canvas > 0.01 * max_intensity)
    if len(nonzero[0]) == 0:
        return {'stamp': np.array([[max_intensity]]), 'size': (1, 1)}

    y_min, y_max = nonzero[0].min(), nonzero[0].max()
    x_min, x_max = nonzero[1].min(), nonzero[1].max()
    stamp = canvas[y_min : y_max + 1, x_min : x_max + 1]

    return {'stamp': stamp, 'size': stamp.shape}


def add_cosmic_rays(
    image,
    n_rays=5,
    length_range=(5, 50),
    width_range=(1, 3),
    intensity_range=(1000, 10000),
    profile='sharp',
):
    """
    Add cosmic ray tracks to an image.

    Cosmic rays are placed at random positions with random orientations.

    :param image: Target image (modified in-place)
    :param n_rays: Number of cosmic rays to add
    :param length_range: (min, max) length in pixels
    :param width_range: (min, max) width in pixels
    :param intensity_range: (min, max) peak intensity in ADU
    :param profile: Intensity profile ('sharp', 'tapered', 'worm')
    :returns: Catalog (astropy Table) of added cosmic rays with columns: x, y, length, width, angle, intensity, type

    """

    height, width = image.shape
    rays = []

    for i in range(n_rays):
        # Random parameters
        length = np.random.uniform(*length_range)
        ray_width = np.random.uniform(*width_range)
        angle = np.random.uniform(0, 180)
        intensity = np.random.uniform(*intensity_range)

        # Random position
        x0 = np.random.uniform(0, width)
        y0 = np.random.uniform(0, height)

        # Create cosmic ray stamp
        cr = create_cosmic_ray(length, ray_width, angle, intensity, profile)
        stamp = cr['stamp']

        # Place stamp on image
        ix, iy = int(np.round(x0)), int(np.round(y0))
        sy, sx = stamp.shape

        # Stamp bounds in image coordinates
        x_start = ix - sx // 2
        y_start = iy - sy // 2

        # Clip to image boundaries
        x_img_start = max(0, x_start)
        y_img_start = max(0, y_start)
        x_img_end = min(width, x_start + sx)
        y_img_end = min(height, y_start + sy)

        # Corresponding stamp coordinates
        x_stamp_start = x_img_start - x_start
        y_stamp_start = y_img_start - y_start
        x_stamp_end = x_stamp_start + (x_img_end - x_img_start)
        y_stamp_end = y_stamp_start + (y_img_end - y_img_start)

        # Add to image
        if x_img_end > x_img_start and y_img_end > y_img_start:
            image[y_img_start:y_img_end, x_img_start:x_img_end] += stamp[
                y_stamp_start:y_stamp_end, x_stamp_start:x_stamp_end
            ]

        rays.append(
            {
                'x': x0,
                'y': y0,
                'length': length,
                'width': ray_width,
                'angle': angle,
                'intensity': intensity,
                'type': 'cosmic_ray',
                'is_real': False,
            }
        )

    return Table(rays)


def add_hot_pixels(
    image, n_pixels=10, intensity_range=(500, 5000), clustering=False, cluster_size=3
):
    """
    Add hot pixels to an image.

    Hot pixels are single bright pixels at random locations, optionally clustered
    to simulate physical detector defects.

    :param image: Target image (modified in-place)
    :param n_pixels: Number of hot pixels to add
    :param intensity_range: (min, max) intensity in ADU
    :param clustering: If True, create clusters of hot pixels
    :param cluster_size: Number of pixels per cluster (if clustering=True)
    :returns: Catalog (astropy Table) of added hot pixels with columns: x, y, intensity, type

    """

    height, width = image.shape
    pixels = []

    if not clustering:
        # Independent hot pixels
        for i in range(n_pixels):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            intensity = np.random.uniform(*intensity_range)

            image[y, x] += intensity

            pixels.append({'x': x, 'y': y, 'intensity': intensity, 'type': 'hot_pixel', 'is_real': False})
    else:
        # Clustered hot pixels — produce exactly n_pixels total
        cluster_size = max(1, cluster_size)
        n_clusters = max(1, int(np.ceil(n_pixels / cluster_size)))
        placed = 0
        for i in range(n_clusters):
            # Cluster center
            cx = np.random.randint(0, width)
            cy = np.random.randint(0, height)

            # Add pixels around center (stop at n_pixels total)
            for j in range(cluster_size):
                if placed >= n_pixels:
                    break

                x = cx + np.random.randint(-2, 3)
                y = cy + np.random.randint(-2, 3)

                # Clip to image boundaries
                x = np.clip(x, 0, width - 1)
                y = np.clip(y, 0, height - 1)

                intensity = np.random.uniform(*intensity_range)
                image[y, x] += intensity

                pixels.append(
                    {'x': x, 'y': y, 'intensity': intensity, 'type': 'hot_pixel', 'is_real': False}
                )
                placed += 1

    return Table(pixels)


def add_bad_columns(
    image, n_columns=2, intensity_range=None, bad_type='dead', orientation='vertical'
):
    """
    Add bad columns or rows to an image.

    Bad columns can be dead (low/zero values), hot (high values), or noisy (high variance).

    :param image: Target image (modified in-place)
    :param n_columns: Number of bad columns/rows to add
    :param intensity_range: (min, max) intensity for 'hot' type. Ignored for 'dead' and 'noisy'.
    :param bad_type: Type of bad column ('dead', 'hot', 'noisy')
    :param orientation: 'vertical' for columns, 'horizontal' for rows
    :returns: Catalog (astropy Table) of bad columns with columns: position, bad_type, orientation, type

    """

    height, width = image.shape
    columns = []

    for i in range(n_columns):
        if orientation == 'vertical':
            # Bad column
            col = np.random.randint(0, width)

            if bad_type == 'dead':
                image[:, col] = 0
            elif bad_type == 'hot':
                if intensity_range is None:
                    intensity_range = (5000, 20000)
                intensity = np.random.uniform(*intensity_range)
                image[:, col] = intensity
            elif bad_type == 'noisy':
                noise = np.random.normal(0, 1000, height)
                image[:, col] += noise
            else:
                raise ValueError(f"Unknown bad_type '{bad_type}'. Use 'dead', 'hot', or 'noisy'")

            columns.append(
                {
                    'position': col,
                    'bad_type': bad_type,
                    'orientation': orientation,
                    'type': 'bad_column',
                    'is_real': False,
                }
            )

        elif orientation == 'horizontal':
            # Bad row
            row = np.random.randint(0, height)

            if bad_type == 'dead':
                image[row, :] = 0
            elif bad_type == 'hot':
                if intensity_range is None:
                    intensity_range = (5000, 20000)
                intensity = np.random.uniform(*intensity_range)
                image[row, :] = intensity
            elif bad_type == 'noisy':
                noise = np.random.normal(0, 1000, width)
                image[row, :] += noise
            else:
                raise ValueError(f"Unknown bad_type '{bad_type}'. Use 'dead', 'hot', or 'noisy'")

            columns.append(
                {
                    'position': row,
                    'bad_type': bad_type,
                    'orientation': orientation,
                    'type': 'bad_row',
                    'is_real': False,
                }
            )

    return Table(columns)


def create_satellite_trail(
    length, width, intensity, angle=None, profile='linear', tumbling=False
):
    """
    Create a satellite trail.

    Satellite trails are long, thin, bright streaks caused by satellites passing through the field.

    :param length: Length of the trail in pixels
    :param width: Width of the trail in pixels
    :param intensity: Peak intensity in ADU
    :param angle: Angle in degrees (0 = horizontal). If None, random.
    :param profile: Transverse profile ('linear', 'gaussian')
    :param tumbling: If True, add intensity variations along the trail (tumbling satellite)
    :returns: Dictionary with 'stamp' (2D array), 'angle' (actual angle used)

    """

    if angle is None:
        angle = np.random.uniform(0, 180)

    # Create horizontal trail on oversized canvas
    canvas_size = int(np.ceil(max(length, width) * 3))
    canvas = np.zeros((canvas_size, canvas_size))

    cx, cy = canvas_size // 2, canvas_size // 2

    track_length = int(np.ceil(length))
    track_width = int(np.ceil(width))

    if profile == 'linear':
        # Rectangular profile
        y_start = cy - track_width // 2
        y_end = y_start + track_width
        x_start = cx - track_length // 2
        x_end = x_start + track_length
        canvas[y_start:y_end, x_start:x_end] = intensity

    elif profile == 'gaussian':
        # Gaussian transverse profile
        x_coords = np.arange(cx - track_length // 2, cx + track_length // 2)
        for x in x_coords:
            for dy in range(-track_width * 2, track_width * 2 + 1):
                y = cy + dy
                if 0 <= y < canvas_size and 0 <= x < canvas_size:
                    gauss = np.exp(-0.5 * (dy / (track_width / 2.355)) ** 2)
                    canvas[y, x] += intensity * gauss

    else:
        raise ValueError(f"Unknown profile '{profile}'. Use 'linear' or 'gaussian'")

    # Add tumbling variations
    if tumbling:
        x_start = cx - track_length // 2
        x_end = x_start + track_length
        variation = 0.5 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, track_length))
        y_start = cy - track_width // 2
        y_end = y_start + track_width
        if x_end - x_start == len(variation):
            canvas[y_start:y_end, x_start:x_end] *= variation[None, :]

    # Rotate
    if angle != 0:
        canvas = rotate(canvas, angle, reshape=False, order=1)

    # Crop to bounding box
    nonzero = np.nonzero(canvas > 0.01 * intensity)
    if len(nonzero[0]) == 0:
        return {'stamp': np.array([[intensity]]), 'angle': angle, 'size': (1, 1)}

    y_min, y_max = nonzero[0].min(), nonzero[0].max()
    x_min, x_max = nonzero[1].min(), nonzero[1].max()
    stamp = canvas[y_min : y_max + 1, x_min : x_max + 1]

    return {'stamp': stamp, 'angle': angle, 'size': stamp.shape}


def add_satellite_trails(
    image,
    n_trails=1,
    length_range=(100, 400),
    width_range=(2, 8),
    intensity_range=(5000, 20000),
    profile='linear',
    tumbling_prob=0.2,
):
    """
    Add satellite trails to an image.

    :param image: Target image (modified in-place)
    :param n_trails: Number of satellite trails to add
    :param length_range: (min, max) length in pixels
    :param width_range: (min, max) width in pixels
    :param intensity_range: (min, max) peak intensity in ADU
    :param profile: Transverse profile ('linear', 'gaussian')
    :param tumbling_prob: Probability of tumbling satellite (intensity variations)
    :returns: Catalog (astropy Table) of added trails

    """

    height, width = image.shape
    trails = []

    for i in range(n_trails):
        length = np.random.uniform(*length_range)
        trail_width = np.random.uniform(*width_range)
        intensity = np.random.uniform(*intensity_range)
        tumbling = np.random.random() < tumbling_prob

        # Random position
        x0 = np.random.uniform(0, width)
        y0 = np.random.uniform(0, height)

        # Create trail
        trail = create_satellite_trail(length, trail_width, intensity, None, profile, tumbling)
        stamp = trail['stamp']
        angle = trail['angle']

        # Place on image
        ix, iy = int(np.round(x0)), int(np.round(y0))
        sy, sx = stamp.shape

        x_start = ix - sx // 2
        y_start = iy - sy // 2

        x_img_start = max(0, x_start)
        y_img_start = max(0, y_start)
        x_img_end = min(width, x_start + sx)
        y_img_end = min(height, y_start + sy)

        x_stamp_start = x_img_start - x_start
        y_stamp_start = y_img_start - y_start
        x_stamp_end = x_stamp_start + (x_img_end - x_img_start)
        y_stamp_end = y_stamp_start + (y_img_end - y_img_start)

        if x_img_end > x_img_start and y_img_end > y_img_start:
            image[y_img_start:y_img_end, x_img_start:x_img_end] += stamp[
                y_stamp_start:y_stamp_end, x_stamp_start:x_stamp_end
            ]

        trails.append(
            {
                'x': x0,
                'y': y0,
                'length': length,
                'width': trail_width,
                'angle': angle,
                'intensity': intensity,
                'tumbling': tumbling,
                'type': 'satellite_trail',
                'is_real': False,
            }
        )

    return Table(trails)


def create_diffraction_spikes(
    size, x0, y0, star_flux, n_spikes=4, spike_length=50, spike_width=2
):
    """
    Create diffraction spikes for a bright star.

    Diffraction spikes are caused by the support structure of telescopes (refractors, SCTs).
    Typically 4 spikes at 45-degree intervals.

    :param size: Size of the stamp
    :param x0: X center (star position)
    :param y0: Y center (star position)
    :param star_flux: Flux of the source star (determines spike intensity)
    :param n_spikes: Number of spikes (typically 4)
    :param spike_length: Length of each spike in pixels
    :param spike_width: Width of each spike in pixels
    :returns: 2D numpy array with diffraction spikes

    """

    stamp = np.zeros((size, size))

    # Spike intensity proportional to star flux with power-law decline
    spike_fraction = 0.01  # 1% of star flux goes into spikes

    for i in range(n_spikes):
        angle = i * (360.0 / n_spikes)  # Evenly spaced angles
        theta = np.radians(angle)

        # Create spike along radial direction
        for r in range(1, int(spike_length)):
            # Radial power law: I(r) ∝ flux / r^2
            intensity = spike_fraction * star_flux / (r**2 + 1)

            # Position along spike
            x = x0 + r * np.cos(theta)
            y = y0 + r * np.sin(theta)

            # Draw spike with finite width
            # for dw in np.linspace(-spike_width / 2, spike_width / 2, int(spike_width) + 1):
            for dw in np.linspace(-np.floor(spike_width / 2), np.floor(spike_width / 2), int(2 * spike_width + 1)):
                xp = x + dw * np.sin(theta)
                yp = y - dw * np.cos(theta)

                ix, iy = int(np.round(xp)), int(np.round(yp))
                if 0 <= ix < size and 0 <= iy < size:
                    stamp[iy, ix] += intensity

    return stamp


def create_optical_ghost(
    size, x0, y0, source_flux, ghost_fraction=0.05, offset=(50, 50), blur_sigma=5.0
):
    """
    Create an optical ghost (reflection artifact).

    Optical ghosts are faint, blurred copies of bright sources caused by reflections
    in the optical system.

    :param size: Minimum size of the stamp (will be expanded if needed to fit ghost)
    :param x0: X center of original source within stamp
    :param y0: Y center of original source within stamp
    :param source_flux: Flux of the original source
    :param ghost_fraction: Fraction of source flux appearing in ghost (typically 0.01-0.1)
    :param offset: (dx, dy) offset of ghost from source in pixels
    :param blur_sigma: Gaussian blur sigma for the ghost
    :returns: Dictionary with 'stamp' (2D array) and 'source_pos' (x, y) giving the
        source position within the stamp (needed to align stamp onto the image)

    """

    # Calculate required stamp size to accommodate both source and ghost.
    # For negative offsets, the ghost lies at coordinates < source position,
    # so we need to shift the origin so all coordinates are non-negative.
    gx = x0 + offset[0]
    gy = y0 + offset[1]

    # Compute origin shift so the leftmost/topmost point is at the margin
    margin = int(blur_sigma * 3)  # 3-sigma margin for Gaussian blur
    shift_x = max(0, margin - int(min(x0, gx)))
    shift_y = max(0, margin - int(min(y0, gy)))

    # Apply shift to both source and ghost coordinates
    x0_s = x0 + shift_x
    y0_s = y0 + shift_y
    gx_s = gx + shift_x
    gy_s = gy + shift_y

    # Stamp must contain both positions plus margin on all sides
    actual_width = max(size, int(max(x0_s, gx_s) + margin + 1))
    actual_height = max(size, int(max(y0_s, gy_s) + margin + 1))

    stamp = np.zeros((actual_height, actual_width))

    # Place point source at ghost position
    ix, iy = int(np.round(gx_s)), int(np.round(gy_s))
    if 0 <= ix < actual_width and 0 <= iy < actual_height:
        stamp[iy, ix] = source_flux * ghost_fraction

        # Blur to simulate defocus
        stamp = gaussian_filter(stamp, sigma=blur_sigma)

    return {'stamp': stamp, 'source_pos': (x0_s, y0_s)}


def add_close_companions_to_catalog(
    catalog,
    fwhm,
    fraction=0.2,
    min_separation_fwhm=1.0,
    max_separation_fwhm=3.0,
    flux_variation=(0.5, 1.5),
    image_shape=None,
    edge=10,
):
    """
    Add close companions to stars in a catalog (before image placement).

    This function creates a new catalog with additional companion stars placed
    near existing stars to simulate crowded fields. Companions are only added
    to sources with type='star'.

    :param catalog: Input catalog (astropy Table) with 'x', 'y', 'flux', 'type' columns
    :param fwhm: FWHM of the PSF in pixels (used to calculate separations)
    :param fraction: Fraction of stars (0-1) that will have companions
    :param min_separation_fwhm: Minimum separation in FWHM units
    :param max_separation_fwhm: Maximum separation in FWHM units
    :param flux_variation: (min, max) flux ratio for companion relative to primary
    :param image_shape: (height, width) tuple for bounds checking, or None to skip
    :param edge: Minimum distance from edges when bounds checking
    :returns: New catalog with companions added (original catalog unchanged)

    """
    from astropy.table import vstack, Table

    # Select only stars for companion addition
    stars = catalog[catalog['type'] == 'star']

    if len(stars) == 0 or fraction <= 0:
        return catalog.copy()

    # Randomly select stars to give companions
    n_companions = int(len(stars) * fraction)
    if n_companions == 0:
        return catalog.copy()

    companion_indices = np.random.choice(len(stars), size=n_companions, replace=False)

    companions_added = []
    min_separation = min_separation_fwhm * fwhm
    max_separation = max_separation_fwhm * fwhm

    for idx in companion_indices:
        source = stars[idx]

        # Random separation and position angle
        separation = np.random.uniform(min_separation, max_separation)
        angle = np.random.uniform(0, 2 * np.pi)

        # Companion position
        comp_x = source['x'] + separation * np.cos(angle)
        comp_y = source['y'] + separation * np.sin(angle)

        # Check if companion is within image bounds
        if image_shape is not None:
            height, width = image_shape
            if not (edge < comp_x < width - edge and edge < comp_y < height - edge):
                continue

        # Companion flux (similar to primary, with variation)
        comp_flux = source['flux'] * np.random.uniform(*flux_variation)

        # Build companion entry matching the catalog structure
        companion = {
            'x': comp_x,
            'y': comp_y,
            'flux': comp_flux,
            'type': 'star',
            'is_real': True,
        }

        # Copy additional columns from source if they exist
        for col in ['fwhm', 'psf_type']:
            if col in source.colnames:
                companion[col] = source[col]

        companions_added.append(companion)

    # Return extended catalog
    if companions_added:
        comp_table = Table(companions_added)
        return vstack([catalog, comp_table])
    else:
        return catalog.copy()


def add_stars(
    image,
    n=100,
    flux_range=(100, 10000),
    fwhm=3.0,
    psf='gaussian',
    beta=2.5,
    edge=0,
    saturation=None,
    diffraction_spikes=False,
    spike_threshold=50000,
    optical_ghosts=False,
    ghost_threshold=100000,
    wcs=None,
):
    """
    Add stars to an image with realistic PSF.

    :param image: Target image (modified in-place)
    :param n: Number of stars to add
    :param flux_range: (min, max) flux range in ADU
    :param fwhm: FWHM of the PSF in pixels (used if psf is a string)
    :param psf: PSF specification - either a string ('gaussian', 'moffat') or a PSF model dictionary (PSFEx format)
    :param beta: Moffat beta parameter (only used if psf='moffat')
    :param edge: Minimum distance from image edges
    :param saturation: Saturation level in ADU
    :param diffraction_spikes: If True, add diffraction spikes to bright stars
    :param spike_threshold: Flux threshold for adding diffraction spikes
    :param optical_ghosts: If True, add optical ghosts to very bright stars
    :param ghost_threshold: Flux threshold for adding optical ghosts
    :param wcs: WCS object for computing RA/Dec
    :returns: Catalog (astropy Table) of added stars

    """

    from . import pipeline
    from . import psf as psf_module

    # Generate random star catalog
    cat = pipeline.make_random_stars(
        shape=image.shape,
        nstars=n,
        minflux=flux_range[0],
        maxflux=flux_range[1],
        edge=edge,
        wcs=wcs,
    )

    # Determine PSF model
    if isinstance(psf, str):
        # Create oversampled PSF model once for all stars
        psf_model = create_psf_model(fwhm=fwhm, psf_type=psf, beta=beta)
        psf_type_name = psf
    elif isinstance(psf, dict):
        # User provided PSF model (e.g., from PSFEx)
        psf_model = psf
        psf_type_name = psf.get('psf_type', 'custom')
        # Extract FWHM from model if not provided
        if 'fwhm' in psf_model:
            fwhm = psf_model['fwhm']
    else:
        raise ValueError(
            "psf parameter must be either a string ('gaussian', 'moffat') or a PSF model dictionary"
        )

    # Add metadata to catalog
    cat['psf_type'] = psf_type_name
    cat['fwhm'] = fwhm
    cat['type'] = 'star'
    cat['is_real'] = True

    # Place stars using PSFEx-compatible placement
    for star in cat:
        x, y, flux = star['x'], star['y'], star['flux']

        # Use psf.place_psf_stamp for fast, accurate placement with sub-pixel positioning
        psf_module.place_psf_stamp(image, psf_model, x, y, flux=flux)

        # Add diffraction spikes if enabled and star is bright enough
        if diffraction_spikes and flux > spike_threshold:
            # Need to create spike stamp and add it manually
            size = int(np.ceil(fwhm * 6)) * 2 + 1
            ix, iy = int(np.round(x)), int(np.round(y))
            dx_sub = x - ix
            dy_sub = y - iy
            x0_spike = size // 2 + dx_sub
            y0_spike = size // 2 + dy_sub

            spikes = create_diffraction_spikes(size, x0_spike, y0_spike, flux)

            # Place spikes on image
            y_grid, x_grid = np.mgrid[0 : spikes.shape[0], 0 : spikes.shape[1]]
            y1, x1 = np.mgrid[0 : spikes.shape[0], 0 : spikes.shape[1]]
            x1 += ix - np.floor(spikes.shape[1] // 2)
            y1 += iy - np.floor(spikes.shape[0] // 2)

            idx = (x1 >= 0) & (x1 < image.shape[1])
            idx &= (y1 >= 0) & (y1 < image.shape[0])

            image[y1[idx].astype(int), x1[idx].astype(int)] += spikes[y_grid[idx], x_grid[idx]]

        # Add optical ghost if enabled and star is very bright
        if optical_ghosts and flux > ghost_threshold:
            size = int(np.ceil(fwhm * 6)) * 2 + 1
            ix, iy = int(np.round(x)), int(np.round(y))
            dx_sub = x - ix
            dy_sub = y - iy
            x0_ghost = size // 2 + dx_sub
            y0_ghost = size // 2 + dy_sub

            ghost_result = create_optical_ghost(size, x0_ghost, y0_ghost, flux)
            ghost = ghost_result['stamp']
            src_x, src_y = ghost_result['source_pos']

            # Place ghost on image
            # The source position within the stamp (src_x, src_y) should align
            # with the star position in the image (ix, iy)
            y_grid, x_grid = np.mgrid[0 : ghost.shape[0], 0 : ghost.shape[1]]
            y1, x1 = np.mgrid[0 : ghost.shape[0], 0 : ghost.shape[1]]

            # Offset so that stamp position (src_x, src_y) maps to image position (ix, iy)
            x1 += ix - int(np.round(src_x))
            y1 += iy - int(np.round(src_y))

            idx = (x1 >= 0) & (x1 < image.shape[1])
            idx &= (y1 >= 0) & (y1 < image.shape[0])

            image[y1[idx].astype(int), x1[idx].astype(int)] += ghost[y_grid[idx], x_grid[idx]]

    # Apply saturation
    if saturation is not None:
        image[image > saturation] = saturation

    return cat


def add_galaxies(
    image,
    n=20,
    flux_range=(500, 5000),
    r_eff_range=(3, 15),
    n_range=(0.5, 4.0),
    ellipticity_range=(0.0, 0.7),
    edge=0,
    wcs=None,
):
    """
    Add galaxies with Sersic profiles to an image.

    :param image: Target image (modified in-place)
    :param n: Number of galaxies to add
    :param flux_range: (min, max) total flux in ADU
    :param r_eff_range: (min, max) effective radius in pixels
    :param n_range: (min, max) Sersic index
    :param ellipticity_range: (min, max) ellipticity
    :param edge: Minimum distance from image edges
    :param wcs: WCS object for computing RA/Dec
    :returns: Catalog (astropy Table) of added galaxies

    """

    height, width = image.shape
    galaxies = []

    for i in range(n):
        # Random parameters
        x = np.random.uniform(edge, width - 1 - edge)
        y = np.random.uniform(edge, height - 1 - edge)
        flux = 10 ** np.random.uniform(np.log10(flux_range[0]), np.log10(flux_range[1]))
        r_eff = np.random.uniform(*r_eff_range)
        sersic_n = np.random.uniform(*n_range)
        ellipticity = np.random.uniform(*ellipticity_range)
        position_angle = np.random.uniform(0, 180)

        # Place galaxy
        place_galaxy(image, x, y, flux, r_eff, sersic_n, ellipticity, position_angle)

        # Catalog entry
        galaxy = {
            'x': x,
            'y': y,
            'flux': flux,
            'r_eff': r_eff,
            'sersic_n': sersic_n,
            'ellipticity': ellipticity,
            'position_angle': position_angle,
            'type': 'galaxy',
            'is_real': True,
        }

        if wcs is not None and wcs.celestial:
            galaxy['ra'], galaxy['dec'] = wcs.all_pix2world(x, y, 0)
        else:
            galaxy['ra'], galaxy['dec'] = np.nan, np.nan

        galaxies.append(galaxy)

    return Table(galaxies)


def simulate_image(
    width,
    height,
    n_stars=100,
    star_flux_range=(100, 10000),
    star_fwhm=3.0,
    star_psf='gaussian',
    star_beta=2.5,
    n_galaxies=20,
    galaxy_flux_range=(500, 5000),
    galaxy_r_eff_range=(3, 15),
    galaxy_n_range=(0.5, 4.0),
    galaxy_ellipticity_range=(0.0, 0.7),
    n_cosmic_rays=5,
    cosmic_ray_length_range=(5, 50),
    cosmic_ray_width_range=(1, 3),
    cosmic_ray_intensity_range=(1000, 10000),
    cosmic_ray_profile='worm',
    n_hot_pixels=10,
    hot_pixel_intensity_range=(500, 5000),
    n_bad_columns=0,
    bad_column_type='dead',
    n_satellites=0,
    satellite_length_range=(100, 400),
    satellite_width_range=(2, 8),
    satellite_intensity_range=(5000, 20000),
    diffraction_spikes=False,
    spike_threshold=50000,
    optical_ghosts=False,
    ghost_threshold=100000,
    add_companions=False,
    companion_fraction=0.2,
    companion_separation_fwhm=(1.0, 3.0),
    companion_flux_ratio=(0.5, 1.5),
    background=1000.0,
    readnoise=10.0,
    gain=1.0,
    edge=10,
    wcs=None,
    return_catalog=True,
    return_masks=False,
    seed=None,
    verbose=False,
):
    """
    Simulate a realistic astronomical image with stars, galaxies, and contaminants.

    This is the high-level API that creates a complete simulated image in one call.

    :param width: Image width in pixels
    :param height: Image height in pixels
    :param n_stars: Number of stars to add
    :param star_flux_range: (min, max) star flux in ADU
    :param star_fwhm: FWHM of stellar PSF in pixels
    :param star_psf: PSF type - either a string ('gaussian', 'moffat') or a PSF model dict
        (from create_psf_model). Use dict to specify aberrated PSFs.
    :param star_beta: Moffat beta parameter (if star_psf='moffat')
    :param n_galaxies: Number of galaxies to add
    :param galaxy_flux_range: (min, max) galaxy flux in ADU
    :param galaxy_r_eff_range: (min, max) effective radius in pixels
    :param galaxy_n_range: (min, max) Sersic index
    :param galaxy_ellipticity_range: (min, max) ellipticity
    :param n_cosmic_rays: Number of cosmic ray tracks
    :param cosmic_ray_length_range: (min, max) cosmic ray length in pixels
    :param cosmic_ray_width_range: (min, max) cosmic ray width in pixels
    :param cosmic_ray_intensity_range: (min, max) cosmic ray peak intensity
    :param cosmic_ray_profile: cosmic ray profile ('sharp', 'tapered', 'worm')
    :param n_hot_pixels: Number of hot pixels
    :param hot_pixel_intensity_range: (min, max) hot pixel intensity
    :param n_bad_columns: Number of bad columns
    :param bad_column_type: Type of bad columns ('dead', 'hot', 'noisy')
    :param n_satellites: Number of satellite trails
    :param satellite_length_range: (min, max) satellite trail length in pixels
    :param satellite_width_range: (min, max) satellite trail width in pixels
    :param satellite_intensity_range: (min, max) satellite trail intensity
    :param diffraction_spikes: Add diffraction spikes to bright stars
    :param spike_threshold: Flux threshold for diffraction spikes
    :param optical_ghosts: Add optical ghosts to very bright stars
    :param ghost_threshold: Flux threshold for optical ghosts
    :param add_companions: If True, add close stellar companions to simulate crowded fields
    :param companion_fraction: Fraction of stars (0-1) that will have companions
    :param companion_separation_fwhm: (min, max) companion separation in FWHM units
    :param companion_flux_ratio: (min, max) companion flux relative to primary star
    :param background: Background level in ADU
    :param readnoise: Read noise in ADU
    :param gain: Detector gain in e-/ADU
    :param edge: Minimum distance from image edges for sources
    :param wcs: WCS object for computing sky coordinates
    :param return_catalog: If True, return catalog of all injected sources
    :param return_masks: If True, return separate masks for each artifact type
    :param seed: Random seed for reproducibility. If set, calls np.random.seed(seed).
    :param verbose: Enable verbose output
    :returns: Dictionary with 'image', 'catalog' (if requested), 'masks' (if requested), 'background', 'noise'

    """

    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if seed is not None:
        np.random.seed(seed)

    log(f"Simulating {width}x{height} image with {n_stars} stars and {n_galaxies} galaxies")

    # Initialize image with background
    image = np.ones((height, width), dtype=float) * background

    # Collect all catalogs
    catalogs = []

    # Add stars
    if n_stars > 0:
        log(f"Adding {n_stars} stars...")

        # Resolve effective FWHM: if star_psf is a dict with its own fwhm, use that
        effective_fwhm = star_fwhm
        if isinstance(star_psf, dict) and 'fwhm' in star_psf:
            effective_fwhm = star_psf['fwhm']

        # Add companions to catalog before placement if requested
        if add_companions:
            # First add the base stars
            cat_stars = add_stars(
                image,
                n=n_stars,
                flux_range=star_flux_range,
                fwhm=star_fwhm,
                psf=star_psf,
                beta=star_beta,
                edge=edge,
                diffraction_spikes=diffraction_spikes,
                spike_threshold=spike_threshold,
                optical_ghosts=optical_ghosts,
                ghost_threshold=ghost_threshold,
                wcs=wcs,
            )

            # Get number of companions before adding
            n_before = len(cat_stars)

            # Add companions to catalog
            cat_stars_with_companions = add_close_companions_to_catalog(
                cat_stars,
                fwhm=effective_fwhm,
                fraction=companion_fraction,
                min_separation_fwhm=companion_separation_fwhm[0],
                max_separation_fwhm=companion_separation_fwhm[1],
                flux_variation=companion_flux_ratio,
                image_shape=(height, width),
                edge=edge,
            )

            n_companions = len(cat_stars_with_companions) - n_before
            if n_companions > 0:
                log(f"  Added {n_companions} close companions")

                # Place only the companion stars (primaries already placed)
                # Determine PSF model for companions
                if isinstance(star_psf, str):
                    psf_model = create_psf_model(fwhm=star_fwhm, psf_type=star_psf, beta=star_beta)
                else:
                    psf_model = star_psf

                # Place companions
                from . import psf as psf_module
                for companion in cat_stars_with_companions[n_before:]:
                    psf_module.place_psf_stamp(
                        image, psf_model, companion['x'], companion['y'],
                        flux=companion['flux']
                    )

            cat_stars = cat_stars_with_companions
        else:
            # No companions - use standard add_stars
            cat_stars = add_stars(
                image,
                n=n_stars,
                flux_range=star_flux_range,
                fwhm=star_fwhm,
                psf=star_psf,
                beta=star_beta,
                edge=edge,
                diffraction_spikes=diffraction_spikes,
                spike_threshold=spike_threshold,
                optical_ghosts=optical_ghosts,
                ghost_threshold=ghost_threshold,
                wcs=wcs,
            )

        catalogs.append(cat_stars)

    # Add galaxies
    if n_galaxies > 0:
        log(f"Adding {n_galaxies} galaxies...")
        cat_galaxies = add_galaxies(
            image,
            n=n_galaxies,
            flux_range=galaxy_flux_range,
            r_eff_range=galaxy_r_eff_range,
            n_range=galaxy_n_range,
            ellipticity_range=galaxy_ellipticity_range,
            edge=edge,
            wcs=wcs,
        )
        catalogs.append(cat_galaxies)

    # Add cosmic rays
    if n_cosmic_rays > 0:
        log(f"Adding {n_cosmic_rays} cosmic rays...")
        cat_cosmic = add_cosmic_rays(
            image,
            n_rays=n_cosmic_rays,
            length_range=cosmic_ray_length_range,
            width_range=cosmic_ray_width_range,
            intensity_range=cosmic_ray_intensity_range,
            profile=cosmic_ray_profile,
        )
        catalogs.append(cat_cosmic)

    # Add hot pixels
    if n_hot_pixels > 0:
        log(f"Adding {n_hot_pixels} hot pixels...")
        cat_hot = add_hot_pixels(
            image, n_pixels=n_hot_pixels, intensity_range=hot_pixel_intensity_range
        )
        catalogs.append(cat_hot)

    # Add bad columns
    if n_bad_columns > 0:
        log(f"Adding {n_bad_columns} bad columns...")
        cat_bad = add_bad_columns(image, n_columns=n_bad_columns, bad_type=bad_column_type)
        catalogs.append(cat_bad)

    # Add satellite trails
    if n_satellites > 0:
        log(f"Adding {n_satellites} satellite trails...")
        cat_satellites = add_satellite_trails(
            image,
            n_trails=n_satellites,
            length_range=satellite_length_range,
            width_range=satellite_width_range,
            intensity_range=satellite_intensity_range,
        )
        catalogs.append(cat_satellites)

    # Apply Poisson noise to entire image (sources + background) in one shot.
    # Physically correct: noise from photon counting on total signal per pixel.
    log("Adding noise...")
    if gain is not None and gain > 0:
        image_e = image * gain
        idx = image_e > 0
        image_e[idx] = np.random.poisson(image_e[idx].astype(np.float64))
        image = image_e / gain

    # Add readnoise (Gaussian, after Poisson — correct physical order)
    noise_map = np.random.normal(0, readnoise, (height, width))
    image += noise_map

    # Prepare output
    result = {'image': image, 'background': np.ones((height, width)) * background, 'noise': noise_map}

    # Combine catalogs
    if return_catalog and len(catalogs) > 0:
        from astropy.table import vstack

        # Stack all catalogs
        combined_cat = vstack(catalogs, join_type='outer')
        result['catalog'] = combined_cat

    if return_masks:
        # Create separate masks for each artifact type
        # This would require tracking which pixels were modified by each artifact
        # For now, return empty dict (can be implemented later if needed)
        result['masks'] = {}

    log("Simulation complete")

    return result


def generate_realbogus_training_data(
    n_images=100,
    image_size=(2048, 2048),
    n_stars_range=(50, 200),
    n_galaxies_range=(10, 50),
    fwhm_range=(1.5, 8.0),
    background_range=(100, 10000),
    n_cosmic_rays_range=(5, 20),
    n_hot_pixels_range=(5, 30),
    n_satellites_range=(0, 2),
    detection_threshold=3.0,
    match_radius=3.0,
    cutout_radius=15,
    augment=True,
    real_source_types=['star'],
    close_pair_fraction=0.2,
    min_separation_fwhm=1.0,
    max_separation_fwhm=3.0,
    aberration_fraction=0.0,
    defocus_range=(0.0, 2.0),
    astigmatism_range=(0.0, 1.5),
    coma_range=(0.0, 1.0),
    readnoise_range=(5.0, 20.0),
    gain_range=(0.5, 2.0),
    asinh_softening=None,
    seed=None,
    verbose=False,
):
    """
    Generate labeled training data for real-bogus classifier.

    This function creates a dataset of labeled cutouts by:
    1. Simulating images with known source positions
    2. Running object detection
    3. Matching detections to truth catalog
    4. Labeling matched detections as 'real', others as 'bogus'
    5. Extracting and preprocessing cutouts
    6. Optionally applying data augmentation

    Flux ranges are automatically calculated for each image based on the background
    noise level to ensure sources are detectable:
    - Minimum flux = detection_threshold × noise (e.g., 3σ for detectability)
    - Maximum flux = 1000 × noise (bright sources)
    This prevents generating sources that are too faint to detect or unrealistically bright.

    Close pairs are automatically added to ensure the classifier learns to accept
    crowded sources. A fraction of stars will have stellar companions added at
    controlled separations (e.g., 1-3 FWHM). Companions are only added to stars,
    not galaxies.

    PSF diversity can be included in training data by setting aberration_fraction > 0.
    A fraction of images will use aberrated PSFs with randomized Zernike coefficients
    (defocus, astigmatism, coma) computed via Fourier optics, helping the classifier
    handle realistic PSF variations from optical aberrations.

    :param n_images: Number of simulated images to generate
    :param image_size: (width, height) of simulated images
    :param n_stars_range: (min, max) number of stars per image
    :param n_galaxies_range: (min, max) number of galaxies per image
    :param fwhm_range: (min, max) FWHM in pixels (varied per image)
    :param background_range: (min, max) background level in ADU
    :param n_cosmic_rays_range: (min, max) cosmic rays per image
    :param n_hot_pixels_range: (min, max) hot pixels per image
    :param n_satellites_range: (min, max) satellite trails per image
    :param detection_threshold: Detection threshold in sigma (also used for min flux calculation)
    :param match_radius: Matching radius in pixels for truth matching
    :param cutout_radius: Cutout radius in pixels
    :param augment: Apply data augmentation (rotations, flips)
    :param real_source_types: List of source types to consider 'real'.
        Default: ['star'] treats only stars as real and galaxies as bogus.
        Use ['star', 'galaxy'] to treat both stars and galaxies as real sources.
    :param close_pair_fraction: Fraction of sources (0-1) that will have close companions added.
        Default: 0.2 (20% of sources get companions)
    :param min_separation_fwhm: Minimum separation for companions in FWHM units.
        Default: 1.0 (1× FWHM)
    :param max_separation_fwhm: Maximum separation for companions in FWHM units.
        Default: 3.0 (3× FWHM)
    :param aberration_fraction: Fraction of images (0-1) with aberrated PSFs.
        Default: 0.0 (all Gaussian). When > 0 and aberration ranges have non-zero
        max values, uses Fourier optics with Zernike polynomials. Otherwise falls
        back to Moffat PSFs as a simpler proxy.
    :param defocus_range: (min, max) defocus aberration in waves (default: (0.0, 2.0))
    :param astigmatism_range: (min, max) astigmatism aberration in waves (default: (0.0, 1.5))
    :param coma_range: (min, max) coma aberration in waves (default: (0.0, 1.0))
    :param readnoise_range: (min, max) read noise in ADU (randomized per image)
    :param gain_range: (min, max) detector gain in e-/ADU (randomized per image)
    :param asinh_softening: Asinh softening in units of background sigma for
        realbogus preprocessing. If None, uses DEFAULT_ASINH_SOFTENING_SIGMA.
    :param seed: Random seed for reproducibility. If set, calls np.random.seed(seed).
    :param verbose: Print progress
    :returns: Dictionary with 'X' (cutouts), 'y' (labels), 'fwhm' (FWHM values), 'metadata'

    """
    from . import photometry
    from . import realbogus

    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if seed is not None:
        np.random.seed(seed)

    log(f"Generating training data from {n_images} simulated images...")

    width, height = image_size

    # Containers for training data
    all_cutouts = []
    all_labels = []
    all_fwhms = []
    all_metadata = []

    for img_idx in range(n_images):
        # Randomize parameters for diversity
        fwhm = np.random.uniform(*fwhm_range)
        background = 10 ** np.random.uniform(np.log10(background_range[0]), np.log10(background_range[1]))
        n_stars = np.random.randint(*n_stars_range)
        n_galaxies = np.random.randint(*n_galaxies_range)
        n_cosmic_rays = np.random.randint(*n_cosmic_rays_range)
        n_hot_pixels = np.random.randint(*n_hot_pixels_range)
        n_satellites = np.random.randint(*n_satellites_range) if n_satellites_range[1] > 0 else 0

        # Randomize readnoise and gain for diversity
        readnoise = np.random.uniform(*readnoise_range)
        gain = np.random.uniform(*gain_range)
        aperture_radius = fwhm
        n_pix = np.pi * aperture_radius**2

        # Noise in aperture (photon noise + readnoise)
        noise_aperture = np.sqrt(n_pix * (background / gain + readnoise**2))

        # Set flux ranges based on S/N ratio
        # min_flux: detectable at threshold (e.g., 3 sigma)
        # max_flux: very bright sources (1000 sigma)
        min_sn = detection_threshold
        max_sn = 1000.0

        star_flux_range = (
            min_sn * noise_aperture,
            max_sn * noise_aperture
        )
        galaxy_flux_range = (
            min_sn * noise_aperture,
            max_sn * noise_aperture
        )

        # Optionally use aberrated PSF for this image to add diversity
        psf_type_str = 'gaussian'
        has_aberration_ranges = (defocus_range[1] > 0 or astigmatism_range[1] > 0
                                 or coma_range[1] > 0)
        if aberration_fraction > 0 and np.random.random() < aberration_fraction:
            if has_aberration_ranges:
                # Use Fourier optics aberrations with random Zernike coefficients
                defocus = np.random.uniform(*defocus_range) if defocus_range[1] > 0 else 0.0
                astig_mag = np.random.uniform(*astigmatism_range) if astigmatism_range[1] > 0 else 0.0
                coma_mag = np.random.uniform(*coma_range) if coma_range[1] > 0 else 0.0

                # Random orientation for directional aberrations
                angle = np.random.uniform(0, 2 * np.pi)
                astigmatism_x = astig_mag * np.cos(angle)
                astigmatism_y = astig_mag * np.sin(angle)
                coma_x = coma_mag * np.cos(angle)
                coma_y = coma_mag * np.sin(angle)

                star_psf = create_psf_model(
                    fwhm=fwhm,
                    psf_type='gaussian',
                    defocus=defocus,
                    astigmatism_x=astigmatism_x,
                    astigmatism_y=astigmatism_y,
                    coma_x=coma_x,
                    coma_y=coma_y,
                )
                psf_type_str = f'aberrated(def={defocus:.1f},ast={astig_mag:.1f},coma={coma_mag:.1f})'
            else:
                # Fall back to Moffat PSF as simpler proxy for non-ideal optics
                beta = np.random.uniform(1.5, 4.5)
                star_psf = create_psf_model(fwhm=fwhm, psf_type='moffat', beta=beta)
                psf_type_str = f'moffat(beta={beta:.1f})'
        else:
            # Use standard Gaussian PSF
            star_psf = 'gaussian'

        log(f"Image {img_idx+1}/{n_images}: FWHM={fwhm:.2f}, BG={background:.1f}, "
            f"stars={n_stars}, gal={n_galaxies}, CR={n_cosmic_rays}, hot={n_hot_pixels}, "
            f"flux_range={star_flux_range[0]:.0f}-{star_flux_range[1]:.0f}, PSF={psf_type_str}")

        # Simulate image with companions (cleaner, unified approach)
        sim = simulate_image(
            width=width,
            height=height,
            n_stars=n_stars,
            star_flux_range=star_flux_range,
            star_fwhm=fwhm,
            star_psf=star_psf,  # Can be string or PSF model dict
            n_galaxies=n_galaxies,
            galaxy_flux_range=galaxy_flux_range,
            n_cosmic_rays=n_cosmic_rays,
            n_hot_pixels=n_hot_pixels,
            n_satellites=n_satellites,
            add_companions=close_pair_fraction > 0,
            companion_fraction=close_pair_fraction,
            companion_separation_fwhm=(min_separation_fwhm, max_separation_fwhm),
            companion_flux_ratio=(0.5, 1.5),
            background=background,
            readnoise=readnoise,
            gain=gain,
            return_catalog=True,
            verbose=False,
        )

        image = sim['image']
        truth_catalog = sim['catalog']

        # Detect objects
        try:
            detected = photometry.get_objects_sep(
                image,
                thresh=detection_threshold,
                aper=fwhm,
                minarea=5,
                verbose=False,
            )
        except Exception as e:
            log(f"Detection failed for image {img_idx+1}: {e}")
            continue

        if len(detected) == 0:
            log(f"No detections in image {img_idx+1}, skipping")
            continue

        # Match detections to truth catalog
        # Real sources: determined by real_source_types parameter
        # Artifacts: cosmic rays, hot pixels, satellites (type='cosmic_ray', etc.)

        # Build filter for real sources based on real_source_types
        real_filter = np.zeros(len(truth_catalog), dtype=bool)
        for source_type in real_source_types:
            real_filter |= (truth_catalog['type'] == source_type)

        truth_real = truth_catalog[real_filter]

        # Simple distance-based matching
        matched_indices = np.full(len(detected), -1, dtype=int)

        for i, det in enumerate(detected):
            dx = truth_real['x'] - det['x']
            dy = truth_real['y'] - det['y']
            dist = np.sqrt(dx**2 + dy**2)

            if len(dist) > 0 and np.min(dist) < match_radius:
                matched_indices[i] = 1  # Real source
            else:
                matched_indices[i] = 0  # Bogus (artifact or spurious)

        # Count matches
        n_real = np.sum(matched_indices == 1)
        n_bogus = np.sum(matched_indices == 0)

        log(f"  Matched: {n_real} real, {n_bogus} bogus from {len(detected)} detections")

        if n_real == 0 and n_bogus == 0:
            continue

        # Extract cutouts
        try:
            cutouts, fwhm_features, valid_indices = realbogus.extract_cutouts(
                detected,
                image,
                bg=background,
                radius=cutout_radius,
                fwhm=fwhm,
                asinh_softening=asinh_softening,
                verbose=False,
            )
        except Exception as e:
            log(f"Cutout extraction failed for image {img_idx+1}: {e}")
            continue

        # Get labels for valid cutouts
        labels = matched_indices[valid_indices]

        # Store data
        all_cutouts.append(cutouts)
        all_labels.append(labels)
        all_fwhms.append(fwhm_features)

        # Metadata for each cutout
        metadata = {
            'image_idx': np.full(len(cutouts), img_idx),
            'fwhm': np.full(len(cutouts), fwhm),
            'background': np.full(len(cutouts), background),
        }
        all_metadata.append(metadata)

    # Concatenate all data
    if len(all_cutouts) == 0:
        raise ValueError("No valid training data generated")

    X = np.concatenate(all_cutouts, axis=0)
    y = np.concatenate(all_labels, axis=0)
    fwhm_feat = np.concatenate(all_fwhms, axis=0)

    log(f"Total samples: {len(X)} ({np.sum(y)} real, {len(y) - np.sum(y)} bogus)")

    # Apply data augmentation
    if augment:
        log("Applying data augmentation...")
        X_aug, y_aug, fwhm_aug = _augment_training_data(X, y, fwhm_feat, verbose=verbose)
        log(f"After augmentation: {len(X_aug)} samples")
    else:
        X_aug = X
        y_aug = y
        fwhm_aug = fwhm_feat

    # Convert labels to binary (ensure proper dtype)
    y_aug = y_aug.astype(np.float32)

    result = {
        'X': X_aug,
        'y': y_aug,
        'fwhm': fwhm_aug,
        'metadata': all_metadata,
    }

    return result


def _augment_training_data(X, y, fwhm, augment_factor=8, verbose=False):
    """
    Apply data augmentation to training cutouts.

    Augmentation strategies:
    - Rotations: 0°, 90°, 180°, 270°
    - Flips: horizontal, vertical
    - Combined: rotations + flips (~8-16× augmentation)

    :param X: Input cutouts (N, H, W, C)
    :param y: Labels (N,)
    :param fwhm: FWHM features (N, 1)
    :param augment_factor: Target augmentation factor
    :param verbose: Print progress
    :returns: (X_aug, y_aug, fwhm_aug) with augmented data
    """
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    n_samples = len(X)
    X_list = [X]
    y_list = [y]
    fwhm_list = [fwhm]

    # Rotation augmentation (90°, 180°, 270°)
    for angle in [90, 180, 270]:
        X_rot = np.rot90(X, k=angle // 90, axes=(1, 2))
        X_list.append(X_rot)
        y_list.append(y)
        fwhm_list.append(fwhm)

    # Flip augmentation
    if augment_factor >= 8:
        # Horizontal flip
        X_flip_h = np.flip(X, axis=2)
        X_list.append(X_flip_h)
        y_list.append(y)
        fwhm_list.append(fwhm)

        # Vertical flip
        X_flip_v = np.flip(X, axis=1)
        X_list.append(X_flip_v)
        y_list.append(y)
        fwhm_list.append(fwhm)

        # Both flips
        X_flip_both = np.flip(np.flip(X, axis=1), axis=2)
        X_list.append(X_flip_both)
        y_list.append(y)
        fwhm_list.append(fwhm)

    # Concatenate
    X_aug = np.concatenate(X_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)
    fwhm_aug = np.concatenate(fwhm_list, axis=0)

    log(f"Augmentation: {n_samples} → {len(X_aug)} samples ({len(X_aug)/n_samples:.1f}× increase)")

    return X_aug, y_aug, fwhm_aug
