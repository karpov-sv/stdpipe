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
from scipy.special import erf
from scipy.ndimage import rotate, gaussian_filter


def create_psf_model(fwhm=3.0, psf_type='gaussian', beta=2.5, size=None, oversampling=4):
    """
    Create an oversampled PSF model compatible with PSFEx format.

    This creates a PSF model that can be used with psf.place_psf_stamp() and other
    STDPipe routines. Uses oversampling for accurate flux conservation and fast
    evaluation with sub-pixel positioning.

    :param fwhm: Full width at half maximum in pixels
    :param psf_type: Type of PSF model ('gaussian' or 'moffat')
    :param beta: Moffat beta parameter (default 2.5, only used for psf_type='moffat')
    :param size: Output stamp size in pixels (after downsampling). If None, automatically sized to capture ~99% of PSF flux (≥8*FWHM)
    :param oversampling: Oversampling factor (default 4)
    :returns: Dictionary with PSFEx-compatible structure containing the PSF model

    """

    # TODO: properly implement oversampling

    # Auto-size to capture full PSF if not specified
    # For Gaussian/Moffat PSF, significant flux extends to ~4 FWHM from center
    # Use 8*FWHM total size to capture ~99% of flux
    if size is None:
        size = int(np.ceil(fwhm * 8))
        if size % 2 == 0:
            size += 1

    # Create PSF directly at target pixel scale (no oversampling)
    # This is simpler and avoids the complex PSFEx width/height/sampling relationship
    # Oversampling will be handled internally by get_psf_stamp when needed
    psf_data_size = size
    psf_width = float(size)
    psf_height = float(size)
    psf_sampling = 1.0  # No oversampling in stored data

    # Center of grid
    center = (psf_data_size - 1) / 2.0

    # Create coordinate grid at target pixel scale
    y, x = np.mgrid[0:psf_data_size, 0:psf_data_size]

    # Distance from center in pixels
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    if psf_type.lower() == 'gaussian':
        # Gaussian PSF
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        psf_data = np.exp(-(r**2) / (2 * sigma**2))

    elif psf_type.lower() == 'moffat':
        # Moffat PSF: I(r) = 1 / (1 + (r/alpha)^2)^beta
        alpha = fwhm / (2 * np.sqrt(2 ** (1.0 / beta) - 1))
        psf_data = 1.0 / (1 + (r / alpha) ** 2) ** beta

    else:
        raise ValueError(
            f"Unknown psf_type '{psf_type}'. Supported types: 'gaussian', 'moffat'"
        )

    # Normalize
    psf_data /= np.sum(psf_data)

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
        'psf_type': psf_type,
        'beta': beta if psf_type.lower() == 'moffat' else None,
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
    image, x0, y0, flux, r_eff=5.0, n=1.0, ellipticity=0.0, position_angle=0.0, gain=None
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
    :param gain: Image gain value. If set, applies Poissonian noise to the galaxy.

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

    # Apply Poissonian noise if gain is provided
    if gain is not None:
        idx = stamp > 0
        stamp[idx] = np.random.poisson(stamp[idx] * gain) / gain

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
        # Tapered track (intensity decreases toward ends)
        y_start = cy - track_width // 2
        y_end = y_start + track_width
        x_start = cx - track_length // 2
        x_end = x_start + track_length
        taper = np.linspace(0.2, 1.0, track_length // 2)
        taper = np.concatenate([taper, taper[::-1]])
        if len(taper) < track_length:
            taper = np.pad(taper, (0, track_length - len(taper)), constant_values=1.0)
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
        # Clustered hot pixels
        n_clusters = max(1, n_pixels // cluster_size)
        for i in range(n_clusters):
            # Cluster center
            cx = np.random.randint(0, width)
            cy = np.random.randint(0, height)

            # Add pixels around center
            for j in range(cluster_size):
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
    :returns: 2D numpy array with optical ghost

    """

    # Calculate required stamp size to accommodate both source and ghost
    # Ghost position relative to source
    gx = x0 + offset[0]
    gy = y0 + offset[1]

    # Expand stamp if needed to contain the ghost (with some margin for blur)
    margin = int(blur_sigma * 3)  # 3-sigma margin for Gaussian blur
    min_width = int(max(x0, gx) + margin + 1)
    min_height = int(max(y0, gy) + margin + 1)

    # Use at least the requested size, but expand if needed
    actual_width = max(size, min_width)
    actual_height = max(size, min_height)

    stamp = np.zeros((actual_height, actual_width))

    # Place point source at ghost position
    if 0 <= gx < actual_width and 0 <= gy < actual_height:
        ix, iy = int(np.round(gx)), int(np.round(gy))
        stamp[iy, ix] = source_flux * ghost_fraction

        # Blur to simulate defocus
        stamp = gaussian_filter(stamp, sigma=blur_sigma)
    else:
        # This should not happen with the size calculation above, but just in case
        # Return empty stamp of requested size
        stamp = np.zeros((size, size))

    return stamp


def add_stars(
    image,
    n=100,
    flux_range=(100, 10000),
    fwhm=3.0,
    psf='gaussian',
    beta=2.5,
    edge=0,
    gain=None,
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
    :param gain: Image gain (for Poisson noise)
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
        psf_module.place_psf_stamp(image, psf_model, x, y, flux=flux, gain=gain)

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

            ghost = create_optical_ghost(size, x0_ghost, y0_ghost, flux)

            # Place ghost on image
            # The source position within the stamp (x0_ghost, y0_ghost) should align
            # with the star position in the image (ix, iy)
            y_grid, x_grid = np.mgrid[0 : ghost.shape[0], 0 : ghost.shape[1]]
            y1, x1 = np.mgrid[0 : ghost.shape[0], 0 : ghost.shape[1]]

            # Offset so that stamp position (x0_ghost, y0_ghost) maps to image position (ix, iy)
            x1 += ix - int(np.round(x0_ghost))
            y1 += iy - int(np.round(y0_ghost))

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
    gain=None,
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
    :param gain: Image gain (for Poisson noise)
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
        place_galaxy(image, x, y, flux, r_eff, sersic_n, ellipticity, position_angle, gain)

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
    background=1000.0,
    readnoise=10.0,
    gain=1.0,
    edge=10,
    wcs=None,
    return_catalog=True,
    return_masks=False,
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
    :param star_psf: PSF type ('gaussian', 'moffat')
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
    :param background: Background level in ADU
    :param readnoise: Read noise in ADU
    :param gain: Detector gain in e-/ADU
    :param edge: Minimum distance from image edges for sources
    :param wcs: WCS object for computing sky coordinates
    :param return_catalog: If True, return catalog of all injected sources
    :param return_masks: If True, return separate masks for each artifact type
    :param verbose: Enable verbose output
    :returns: Dictionary with 'image', 'catalog' (if requested), 'masks' (if requested), 'background', 'noise'

    """

    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    log(f"Simulating {width}x{height} image with {n_stars} stars and {n_galaxies} galaxies")

    # Initialize image with background
    image = np.ones((height, width), dtype=float) * background

    # Collect all catalogs
    catalogs = []

    # Add stars
    if n_stars > 0:
        log(f"Adding {n_stars} stars...")
        cat_stars = add_stars(
            image,
            n=n_stars,
            flux_range=star_flux_range,
            fwhm=star_fwhm,
            psf=star_psf,
            beta=star_beta,
            edge=edge,
            gain=gain,
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
            gain=gain,
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

    # Add noise
    log("Adding noise...")
    noise_map = np.random.normal(0, readnoise, (height, width))
    image += noise_map

    # Poisson noise from sources (if gain is set, already applied to individual sources)
    # Add Poisson noise from background
    if gain is not None:
        poisson_noise = np.random.poisson(background * gain, (height, width)) / gain - background
        image += poisson_noise

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

    :param n_images: Number of simulated images to generate
    :param image_size: (width, height) of simulated images
    :param n_stars_range: (min, max) number of stars per image
    :param n_galaxies_range: (min, max) number of galaxies per image
    :param fwhm_range: (min, max) FWHM in pixels (varied per image)
    :param background_range: (min, max) background level in ADU
    :param n_cosmic_rays_range: (min, max) cosmic rays per image
    :param n_hot_pixels_range: (min, max) hot pixels per image
    :param n_satellites_range: (min, max) satellite trails per image
    :param detection_threshold: Detection threshold in sigma
    :param match_radius: Matching radius in pixels for truth matching
    :param cutout_radius: Cutout radius in pixels
    :param augment: Apply data augmentation (rotations, flips)
    :param verbose: Print progress
    :returns: Dictionary with 'X' (cutouts), 'y' (labels), 'fwhm' (FWHM values), 'metadata'

    """
    from . import photometry
    from . import realbogus

    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

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

        log(f"Image {img_idx+1}/{n_images}: FWHM={fwhm:.2f}, BG={background:.1f}, "
            f"stars={n_stars}, gal={n_galaxies}, CR={n_cosmic_rays}, hot={n_hot_pixels}")

        # Simulate image
        sim = simulate_image(
            width=width,
            height=height,
            n_stars=n_stars,
            star_fwhm=fwhm,
            n_galaxies=n_galaxies,
            n_cosmic_rays=n_cosmic_rays,
            n_hot_pixels=n_hot_pixels,
            n_satellites=n_satellites,
            background=background,
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
                aper=2.5 * fwhm,
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
        # Real sources: stars and galaxies (type='star' or type='galaxy')
        # Artifacts: cosmic rays, hot pixels, satellites (type='cosmic_ray', etc.)

        truth_real = truth_catalog[
            (truth_catalog['type'] == 'star') | (truth_catalog['type'] == 'galaxy')
        ]

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

    log(f"Total samples: {len(X)} ({np.sum(y)} real, {np.sum(~y)} bogus)")

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


def _augment_training_data(X, y, fwhm, augment_factor=4, verbose=False):
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
