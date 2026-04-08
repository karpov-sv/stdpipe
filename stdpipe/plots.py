import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rcParams
import builtins

from astropy.stats import mad_std
from astropy.visualization import simple_norm, ImageNormalize
from astropy.visualization.stretch import HistEqStretch
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import binned_statistic_2d
from scipy.spatial import Voronoi

from . import photometry

# Optional powerbin import
try:
    import powerbin

    HAS_POWERBIN = True
except ImportError:
    HAS_POWERBIN = False


def colorbar(obj=None, ax=None, size="5%", pad=0.1):
    should_restore = False

    if ax is None and obj is not None:
        ax = obj.axes
    elif ax is None:
        ax = plt.gca()
        # should_restore = True

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)

    ax.get_figure().colorbar(obj, cax=cax)

    # if should_restore:
    ax.get_figure().sca(ax)


def imshow(
    image,
    qq=None,
    mask=None,
    show_colorbar=True,
    show_axis=True,
    stretch='linear',
    r0=None,
    ax=None,
    max_plot_size=4096,
    fast=True,
    xlim=None,
    ylim=None,
    **kwargs,
):
    """Display a 2D image with percentile-based intensity scaling.

    Parameters
    ----------
    image : ndarray
        2D array to display.
    qq : list of float, optional
        Two-element ``[low, high]`` percentile range for intensity normalization.
        Default is ``[0.5, 99.5]``. Overridden by explicit ``vmin`` / ``vmax``.
    mask : ndarray of bool, optional
        Boolean mask; masked pixels are excluded from intensity normalization.
    show_colorbar : bool, optional
        If True, display a colorbar alongside the image.
    show_axis : bool, optional
        If True, display axis ticks and labels.
    stretch : str, optional
        Intensity stretch: ``'linear'``, ``'log'``, ``'asinh'``, ``'histeq'``,
        or any stretch supported by Astropy visualization.
    r0 : float, optional
        Gaussian smoothing sigma in pixels applied before display.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; defaults to the current axes.
    max_plot_size : int or None, optional
        Images larger than this (in pixels) are downscaled for rendering.
        Set to None to disable. Default is 4096.
    fast : bool, optional
        If True, use faster approximate methods for large images (subsampled
        percentiles, float32, FFT convolution).
    xlim : tuple of float, optional
        ``(xmin, xmax)`` region selection in original image coordinates.
    ylim : tuple of float, optional
        ``(ymin, ymax)`` region selection in original image coordinates.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.imshow`.
    """
    if ax is None:
        ax = plt.gca()

    # Store original shape for coordinate system
    orig_shape = image.shape

    # STEP 0: Region selection (FIRST, before any processing)
    if xlim is not None or ylim is not None:
        x0 = int(np.floor(xlim[0])) if xlim is not None else 0
        x1 = int(np.ceil(xlim[1])) + 1 if xlim is not None else orig_shape[1]
        y0 = int(np.floor(ylim[0])) if ylim is not None else 0
        y1 = int(np.ceil(ylim[1])) + 1 if ylim is not None else orig_shape[0]

        # Clip to image bounds
        x0 = max(0, min(x0, orig_shape[1]))
        x1 = max(0, min(x1, orig_shape[1]))
        y0 = max(0, min(y0, orig_shape[0]))
        y1 = max(0, min(y1, orig_shape[0]))

        # Extract region at full resolution
        image = image[y0:y1, x0:x1]
        if mask is not None:
            mask = mask[y0:y1, x0:x1]

        region_offset = (x0, y0)
        region_shape = (y1 - y0, x1 - x0)
    else:
        region_offset = (0, 0)
        region_shape = orig_shape

    # OPTIMIZATION 1: Downscale large images for matplotlib
    if fast and max_plot_size is not None and max(image.shape) > max_plot_size:
        scale = max_plot_size / max(image.shape)
        from scipy.ndimage import zoom

        image = zoom(image, scale, order=1)  # bilinear interpolation
        if mask is not None:
            mask = zoom(mask.astype(np.uint8), scale, order=0).astype(bool)

    # OPTIMIZATION 2: Use float32 instead of float64 for memory efficiency
    if r0 is not None and r0 > 0:
        # Need to convert for smoothing operation
        image = image.astype(np.float32)
    elif not np.issubdtype(image.dtype, np.floating):
        # For integer images, convert for proper display
        image = image.astype(np.float32)
    else:
        # Already floating point, use as-is
        image = np.asarray(image)

    # OPTIMIZATION 3: Fast finite check for large images
    if fast and image.size > 10_000_000:
        # Quick check: are there ANY non-finite values in a sample?
        if not np.all(np.isfinite(image.flat[:10000])):
            good_idx = np.isfinite(image)
        else:
            good_idx = np.ones(image.shape, dtype=bool)
    else:
        good_idx = np.isfinite(image)

    if mask is not None:
        good_idx &= ~mask

    if np.sum(good_idx):
        # OPTIMIZATION 4: FFT convolution for large images
        if r0 is not None and r0 > 0:
            kernel = Gaussian2DKernel(r0)
            # Use FFT convolution for large images (much faster)
            if fast and image.size > 1_000_000:
                from astropy.convolution import convolve_fft

                image = convolve_fft(
                    image, kernel, mask=mask, boundary='extend', nan_treatment='fill', fill_value=0
                )
            else:
                image = convolve(image, kernel, mask=mask, boundary='extend')

        if qq is None and 'vmin' not in kwargs and 'vmax' not in kwargs:
            # Sane defaults for quantiles if no manual limits provided
            qq = [0.5, 99.5]

        # OPTIMIZATION 5: Subsample for percentiles on large images
        if qq is not None:
            # Presence of qq quantiles overwrites vmin/vmax even if they are present
            max_pixels = 1_000_000
            n_good = np.sum(good_idx)
            if fast and n_good > max_pixels:
                # Random subsample of good pixels for percentile calculation
                good_coords = np.where(good_idx)
                sample_idx = np.random.choice(len(good_coords[0]), max_pixels, replace=False)
                sample_values = image[good_coords[0][sample_idx], good_coords[1][sample_idx]]
                kwargs['vmin'], kwargs['vmax'] = np.percentile(sample_values, qq)
            else:
                kwargs['vmin'], kwargs['vmax'] = np.percentile(image[good_idx], qq)

        if not 'interpolation' in kwargs:
            # Rough heuristic to choose interpolation method based on image dimensions
            if image.shape[0] < 300 and image.shape[1] < 300:
                kwargs['interpolation'] = 'nearest'
            else:
                kwargs['interpolation'] = 'bicubic'

        # OPTIMIZATION 6: Optimize stretch (especially histeq mode)
        if stretch and stretch != 'linear':
            if stretch == 'histeq':
                # Use only valid pixels for histogram, avoid creating multiple copies
                if 'vmin' in kwargs or 'vmax' in kwargs:
                    vmin = kwargs.get('vmin', -np.inf)
                    vmax = kwargs.get('vmax', np.inf)
                    valid_mask = (image >= vmin) & (image <= vmax) & good_idx
                    data = image[valid_mask]
                else:
                    data = image[good_idx]

                kwargs['norm'] = ImageNormalize(
                    stretch=HistEqStretch(data),
                    vmin=kwargs.pop('vmin', None),
                    vmax=kwargs.pop('vmax', None),
                )
            else:
                kwargs['norm'] = simple_norm(
                    image,
                    stretch,
                    min_cut=kwargs.pop('vmin', None),
                    max_cut=kwargs.pop('vmax', None),
                    power=2,
                )

    # CRITICAL: Preserve coordinate system (including region offset)
    if 'extent' not in kwargs:
        x0_extent = region_offset[0] - 0.5
        y0_extent = region_offset[1] - 0.5
        x1_extent = x0_extent + region_shape[1]
        y1_extent = y0_extent + region_shape[0]

        # extent = [left, right, bottom, top]
        # For origin='upper' (default): y increases downward, so bottom=y1, top=y0
        # For origin='lower': y increases upward, so bottom=y0, top=y1
        origin = kwargs.get('origin', rcParams.get('image.origin', 'upper'))
        if origin == 'lower':
            kwargs['extent'] = [x0_extent, x1_extent, y0_extent, y1_extent]
        else:
            kwargs['extent'] = [x0_extent, x1_extent, y1_extent, y0_extent]

    img = ax.imshow(image, **kwargs)
    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_axis_on()
    if show_colorbar:
        colorbar(img, ax=ax)
    else:
        # Mimic the extension of scaling limits if they are equal
        clim = img.get_clim()

        if clim[0] == clim[1]:
            img.set_clim(clim[0] - 0.1, clim[1] + 0.1)

    return img


def binned_map(
    x,
    y,
    value,
    bins=16,
    statistic='mean',
    qq=[0.5, 97.5],
    color=None,
    show_colorbar=True,
    show_axis=True,
    show_dots=False,
    ax=None,
    range=None,
    **kwargs,
):
    """Plot statistical estimators binned onto a regular grid.

    Parameters
    ----------
    x : array_like
        Abscissae of the data points.
    y : array_like
        Ordinates of the data points.
    value : array_like
        Values to aggregate.
    bins : int, optional
        Number of bins per axis.
    statistic : str or callable, optional
        Aggregation statistic: ``'mean'``, ``'median'``, or a callable.
    qq : list of float, optional
        ``[low, high]`` percentile range for color scaling. Default ``[0.5, 97.5]``.
        Overridden by explicit ``vmin`` / ``vmax``.
    color : color, optional
        Color used for the data-point overlay (requires ``show_dots=True``).
    show_colorbar : bool, optional
        If True, display a colorbar.
    show_axis : bool, optional
        If True, display axis ticks and labels.
    show_dots : bool, optional
        If True, overlay the raw data point positions.
    range : list of list, optional
        Data range ``[[xmin, xmax], [ymin, ymax]]``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; defaults to the current axes.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.imshow`.
    """
    gmag0, xe, ye, binnumbers = binned_statistic_2d(
        x, y, value, bins=bins, statistic=statistic, range=range
    )

    vmin1, vmax1 = np.percentile(gmag0[np.isfinite(gmag0)], qq)
    if not 'vmin' in kwargs:
        kwargs['vmin'] = vmin1
    if not 'vmax' in kwargs:
        kwargs['vmax'] = vmax1

    if ax is None:
        ax = plt.gca()

    if not 'aspect' in kwargs:
        kwargs['aspect'] = 'auto'

    im = ax.imshow(
        gmag0.T,
        origin='lower',
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        interpolation='nearest',
        **kwargs,
    )
    if show_colorbar:
        colorbar(im, ax=ax)

    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_axis_on()

    if show_dots:
        ax.set_autoscale_on(False)
        ax.plot(x, y, '.', color=color, alpha=0.3)


def _kdtree_adaptive_bins(x, y, target_count, data_range=None):
    """
    Create adaptive bins using recursive K-D tree splitting.

    Recursively splits the data region into rectangular bins until each
    bin contains approximately `target_count` points.

    Parameters
    ----------
    x, y : array-like
        Coordinates of data points
    target_count : int
        Target number of points per bin
    data_range : list, optional
        [[xmin, xmax], [ymin, ymax]] range for binning

    Returns
    -------
    list of tuples
        Each tuple contains (xmin, xmax, ymin, ymax, indices) where indices
        is a boolean array selecting points in that bin.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if data_range is not None:
        xmin, xmax = data_range[0]
        ymin, ymax = data_range[1]
    else:
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)

    bins = []

    def _split_recursive(idx, xmin, xmax, ymin, ymax, depth=0):
        """Recursively split region until target count reached."""
        count = np.sum(idx)

        # Stop splitting if we have few enough points or region is too small
        if count <= target_count * 1.5 or count < 3:
            if count > 0:
                bins.append((xmin, xmax, ymin, ymax, idx.copy()))
            return

        # Alternate splitting direction based on depth, prefer longer dimension
        width = xmax - xmin
        height = ymax - ymin

        if (depth % 2 == 0 and width >= height) or (depth % 2 == 1 and width > height):
            # Split in x
            x_in_region = x[idx]
            mid = np.median(x_in_region)
            left_idx = idx & (x <= mid)
            right_idx = idx & (x > mid)

            if np.sum(left_idx) > 0 and np.sum(right_idx) > 0:
                _split_recursive(left_idx, xmin, mid, ymin, ymax, depth + 1)
                _split_recursive(right_idx, mid, xmax, ymin, ymax, depth + 1)
            else:
                bins.append((xmin, xmax, ymin, ymax, idx.copy()))
        else:
            # Split in y
            y_in_region = y[idx]
            mid = np.median(y_in_region)
            bottom_idx = idx & (y <= mid)
            top_idx = idx & (y > mid)

            if np.sum(bottom_idx) > 0 and np.sum(top_idx) > 0:
                _split_recursive(bottom_idx, xmin, xmax, ymin, mid, depth + 1)
                _split_recursive(top_idx, xmin, xmax, mid, ymax, depth + 1)
            else:
                bins.append((xmin, xmax, ymin, ymax, idx.copy()))

    # Start with all points in the data range
    initial_idx = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    _split_recursive(initial_idx, xmin, xmax, ymin, ymax)

    return bins


def _powerbin_adaptive_bins(x, y, target_count, data_range=None, verbose=False):
    """
    Create adaptive bins using PowerBin's centroidal power diagrams.

    Parameters
    ----------
    x, y : array-like
        Coordinates of data points
    target_count : int
        Target number of points per bin
    data_range : list, optional
        [[xmin, xmax], [ymin, ymax]] range for binning
    verbose : bool, optional
        Whether to enable PowerBin verbose output

    Returns
    -------
    bin_numbers : ndarray
        Bin assignment for each point
    centroids : ndarray
        Centroids of each bin (N_bins x 2)
    """
    if not HAS_POWERBIN:
        raise ImportError("powerbin package is required for this method")

    from powerbin import PowerBin

    x = np.asarray(x)
    y = np.asarray(y)

    if data_range is not None:
        xmin, xmax = data_range[0]
        ymin, ymax = data_range[1]
        mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        x_use = x[mask]
        y_use = y[mask]
    else:
        x_use = x
        y_use = y
        mask = np.ones(len(x), dtype=bool)

    n_points = len(x_use)
    if n_points < 4:
        # Not enough points for PowerBin
        raise ValueError("Need at least 4 points for PowerBin")

    # PowerBin expects xy coordinates and capacity per pixel
    # For count-based binning, each pixel has capacity 1
    xy = np.column_stack([x_use, y_use])
    capacity = np.ones(n_points)  # Each point contributes 1 to capacity

    # Run PowerBin - may fail for scattered (non-gridded) point data
    try:
        pb = PowerBin(
            xy,
            capacity_spec=capacity,
            target_capacity=target_count,
            verbose=1 if verbose else 0,
            maxiter=500,
        )
    except (IndexError, ValueError) as e:
        raise RuntimeError(f"PowerBin failed (may not work with scattered point data): {e}")

    bin_numbers_subset = pb.bin_num  # Note: attribute is bin_num, not bin_number
    centroids = pb.xybin  # 2D array of bin centroids

    # Map back to full array if range was specified
    if data_range is not None:
        bin_numbers = np.full(len(x), -1)
        bin_numbers[mask] = bin_numbers_subset
    else:
        bin_numbers = bin_numbers_subset

    return bin_numbers, centroids


def adaptive_binned_map(
    x,
    y,
    value,
    target_count=50,
    target_sn=None,
    err=None,
    statistic='mean',
    method='auto',
    qq=[0.5, 97.5],
    color=None,
    show_colorbar=True,
    show_axis=True,
    show_dots=False,
    show_edges=True,
    ax=None,
    range=None,
    verbose=False,
    **kwargs,
):
    """Plot statistical estimators with adaptive (density-based) binning.

    Bins have variable sizes: sparse regions get larger bins, dense regions
    get finer resolution.  Bins target approximately ``target_count`` points
    each, or a specified ``target_sn`` S/N ratio.

    Parameters
    ----------
    x : array_like
        Abscissae of the data points.
    y : array_like
        Ordinates of the data points.
    value : array_like
        Values to aggregate.
    target_count : int, optional
        Target number of points per bin.
    target_sn : float, optional
        Target S/N per bin (alternative to ``target_count``).
    err : array_like, optional
        Per-point value errors; required when ``target_sn`` is set.
    statistic : str or callable, optional
        Aggregation: ``'mean'``, ``'median'``, ``'std'``, ``'count'``, or callable.
    method : str, optional
        Binning algorithm: ``'auto'``, ``'powerbin'``, or ``'kdtree'``.
    qq : list of float, optional
        ``[low, high]`` percentile range for color scaling. Default ``[0.5, 97.5]``.
    color : color, optional
        Color for the data-point overlay (requires ``show_dots=True``).
    show_colorbar : bool, optional
        If True, display a colorbar.
    show_axis : bool, optional
        If True, display axis ticks and labels.
    show_dots : bool, optional
        If True, overlay the raw data point positions.
    show_edges : bool, optional
        If True, draw bin boundaries.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; defaults to the current axes.
    range : list of list, optional
        Data range ``[[xmin, xmax], [ymin, ymax]]``.
    verbose : bool, optional
        If True, enable verbose output from the binning backend.
    **kwargs
        Additional keyword arguments passed to matplotlib.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    value = np.asarray(value)

    if ax is None:
        ax = plt.gca()

    # Filter non-finite data early
    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(value)
    if err is not None:
        err = np.asarray(err)
        finite_mask &= np.isfinite(err)
    x = x[finite_mask]
    y = y[finite_mask]
    value = value[finite_mask]
    if err is not None:
        err = err[finite_mask]

    if range is not None:
        xmin, xmax = range[0]
        ymin, ymax = range[1]
        range_mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        x = x[range_mask]
        y = y[range_mask]
        value = value[range_mask]
        if err is not None:
            err = err[range_mask]

    if len(x) == 0:
        raise ValueError("No finite data points within the specified range.")

    # Determine method
    if method == 'auto':
        method = 'powerbin' if HAS_POWERBIN else 'kdtree'

    # Calculate target_count from target_sn if provided
    if target_sn is not None:
        if err is None:
            raise ValueError("noise parameter required when using target_sn")
        # For S/N targeting, estimate how many points needed
        # S/N ~ sqrt(N) * mean_signal / mean_noise for Poisson
        valid_sn = err > 0
        if np.any(valid_sn):
            mean_sn = np.nanmean(np.abs(value[valid_sn]) / err[valid_sn])
        else:
            mean_sn = np.nan
        if mean_sn > 0:
            target_count = max(3, int((target_sn / mean_sn) ** 2))
        else:
            target_count = 50  # fallback

    # Get statistic function
    if callable(statistic):
        stat_func = statistic
    elif statistic == 'mean':
        stat_func = np.nanmean
    elif statistic == 'median':
        stat_func = np.nanmedian
    elif statistic == 'std':
        stat_func = np.nanstd
    elif statistic == 'count':
        stat_func = lambda v: len(v)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Compute adaptive bins
    powerbin_success = False
    if range is None:
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
    else:
        xmin, xmax = range[0]
        ymin, ymax = range[1]

    if xmin == xmax or ymin == ymax:
        method = 'kdtree'

    if method == 'powerbin':
        if not HAS_POWERBIN:
            raise ImportError("powerbin package required. Install with: pip install powerbin")

        try:
            bin_numbers, centroids = _powerbin_adaptive_bins(
                x, y, target_count, range, verbose=verbose
            )
            powerbin_success = True
        except RuntimeError:
            # PowerBin failed (common with scattered point data), fall back to kdtree
            pass

        if powerbin_success:
            # Compute statistic per bin
            bin_values = np.full(len(centroids), np.nan)
            for b in builtins.range(len(centroids)):
                mask = bin_numbers == b
                if np.any(mask):
                    bin_values[b] = stat_func(value[mask])

            # Create Voronoi diagram for visualization
            if len(centroids) >= 4:
                # Add corner points to bound the Voronoi diagram
                # Pad to ensure all regions are finite
                pad_x = (xmax - xmin) * 0.1
                pad_y = (ymax - ymin) * 0.1
                corners = np.array(
                    [
                        [xmin - pad_x, ymin - pad_y],
                        [xmin - pad_x, ymax + pad_y],
                        [xmax + pad_x, ymin - pad_y],
                        [xmax + pad_x, ymax + pad_y],
                    ]
                )
                all_centroids = np.vstack([centroids, corners])
                vor = Voronoi(all_centroids)

                # Compute vmin/vmax
                finite_vals = bin_values[np.isfinite(bin_values)]
                if len(finite_vals) > 0:
                    vmin1, vmax1 = np.nanpercentile(finite_vals, qq)
                else:
                    vmin1, vmax1 = 0, 1

                if 'vmin' not in kwargs:
                    kwargs['vmin'] = vmin1
                if 'vmax' not in kwargs:
                    kwargs['vmax'] = vmax1

                # Get colormap
                cmap = kwargs.pop('cmap', 'viridis')
                if isinstance(cmap, str):
                    cmap = plt.get_cmap(cmap)

                norm = plt.Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax'])

                # Draw Voronoi cells
                from matplotlib.collections import PolyCollection

                polygons = []
                colors = []
                for i, val in enumerate(bin_values):
                    region_idx = vor.point_region[i]
                    region = vor.regions[region_idx]
                    if -1 not in region and len(region) > 0:
                        polygon = [vor.vertices[j] for j in region]
                        # Clip to data range
                        polygon = np.array(polygon)
                        polygon[:, 0] = np.clip(polygon[:, 0], xmin, xmax)
                        polygon[:, 1] = np.clip(polygon[:, 1], ymin, ymax)
                        polygons.append(polygon)
                        colors.append(cmap(norm(val)) if np.isfinite(val) else (0.8, 0.8, 0.8, 1))

                pc = PolyCollection(
                    polygons,
                    facecolors=colors,
                    edgecolors=kwargs.pop('edgecolors', 'black') if show_edges else 'none',
                    linewidths=kwargs.pop('linewidths', 0.5) if show_edges else 0,
                )
                ax.add_collection(pc)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                if show_colorbar:
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    # ax.get_figure().colorbar(sm, ax=ax)
                    colorbar(sm, ax=ax)

            else:
                # Too few bins, will fall back to kdtree
                powerbin_success = False

    if method == 'kdtree' or (method == 'powerbin' and not powerbin_success):
        bins_list = _kdtree_adaptive_bins(x, y, target_count, range)

        # Compute statistic per bin
        bin_values = []
        for xmin_b, xmax_b, ymin_b, ymax_b, idx in bins_list:
            bin_values.append(stat_func(value[idx]))
        bin_values = np.array(bin_values)

        # Compute vmin/vmax
        finite_vals = bin_values[np.isfinite(bin_values)]
        if len(finite_vals) > 0:
            vmin1, vmax1 = np.nanpercentile(finite_vals, qq)
        else:
            vmin1, vmax1 = 0, 1

        if 'vmin' not in kwargs:
            kwargs['vmin'] = vmin1
        if 'vmax' not in kwargs:
            kwargs['vmax'] = vmax1

        # Get colormap
        cmap = kwargs.pop('cmap', 'viridis')
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        norm = plt.Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax'])

        # Draw rectangles
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection

        patches = []
        colors = []
        for i, (xmin_b, xmax_b, ymin_b, ymax_b, idx) in enumerate(bins_list):
            rect = Rectangle(
                (xmin_b, ymin_b),
                xmax_b - xmin_b,
                ymax_b - ymin_b,
            )
            patches.append(rect)
            val = bin_values[i]
            colors.append(cmap(norm(val)) if np.isfinite(val) else (0.8, 0.8, 0.8, 1))

        pc = PatchCollection(
            patches,
            facecolors=colors,
            edgecolors='black' if show_edges else 'none',
            linewidths=0.5 if show_edges else 0,
        )
        ax.add_collection(pc)

        # Set axis limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        if show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # ax.get_figure().colorbar(sm, ax=ax)
            colorbar(sm, ax=ax)

    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_axis_on()

    if show_dots:
        ax.set_autoscale_on(False)
        ax.plot(x, y, '.', color=color, alpha=0.3)

    ax.set_aspect(kwargs.get('aspect', 'auto'))
    # if 'aspect' not in kwargs:
    #     ax.set_aspect('auto')


def plot_cutout(
    cutout,
    planes=['image', 'template', 'diff', 'mask'],
    fig=None,
    axs=None,
    mark_x=None,
    mark_y=None,
    mark_r=5.0,
    mark_r2=None,
    mark_r3=None,
    mark_color='red',
    mark_lw=2,
    mark_ra=None,
    mark_dec=None,
    r0=None,
    show_title=True,
    title=None,
    additional_title=None,
    **kwargs,
):
    """Display image planes from a cutout structure in a single row.

    Parameters
    ----------
    cutout : dict
        Cutout structure as returned by :func:`stdpipe.cutouts.get_cutout`.
    planes : list of str, optional
        Names of cutout planes to show (in order).
    fig : matplotlib.figure.Figure, optional
        Figure to draw into; a new figure is created if not provided.
    axs : list of matplotlib.axes.Axes, optional
        Axes to draw into; must be the same length as ``planes``.
    mark_x : float, optional
        X coordinate (in cutout pixels) of the circular overlay mark.
    mark_y : float, optional
        Y coordinate (in cutout pixels) of the circular overlay mark.
    mark_r : float, optional
        Radius of the overlay mark in pixels.
    mark_r2 : float, optional
        Radius of a secondary dashed overlay circle.
    mark_r3 : float, optional
        Radius of a tertiary dashed overlay circle.
    mark_color : color, optional
        Color of the overlay mark.
    mark_lw : float, optional
        Line width of the overlay mark.
    mark_ra : float, optional
        RA of the overlay mark; overrides ``mark_x`` / ``mark_y``.
    mark_dec : float, optional
        Dec of the overlay mark; overrides ``mark_x`` / ``mark_y``.
    r0 : float, optional
        Gaussian smoothing sigma applied to ``image``, ``template``, and ``diff`` planes.
    show_title : bool, optional
        If True (default), display a title above the cutout row.
    title : str, optional
        Title text. Auto-generated from cutout metadata if not provided.
    additional_title : str, optional
        Text appended to the auto-generated title.
    **kwargs
        Additional keyword arguments passed to :func:`imshow` for each plane.
    """

    curplot = 1

    nplots = len([_ for _ in planes if _ in cutout])

    if fig is None:
        fig = plt.figure(figsize=[nplots * 4, 4 + 1.0], dpi=75, tight_layout=True)

    if axs is not None:
        if not len(axs) == len(planes):
            raise ValueError('Number of axes must be same as number of cutouts')

    for ii, name in enumerate(planes):
        if name in cutout and cutout[name] is not None:
            if axs is not None:
                ax = axs[ii]
            else:
                ax = fig.add_subplot(1, nplots, curplot)
            curplot += 1

            params = {
                'stretch': 'asinh' if name in ['image', 'template', 'convolved'] else 'linear',
                'r0': r0 if name in ['image', 'template', 'diff'] else None,
                # 'qq': [0.5, 100] if name in ['image', 'template', 'convolved'] else [0.5, 99.5],
                'cmap': 'Blues_r',
                'show_colorbar': False,
                'show_axis': False,
            }

            params.update(kwargs)

            imshow(cutout[name], ax=ax, **params)
            ax.set_title(name.upper())

            if mark_ra is not None and mark_dec is not None and cutout.get('wcs'):
                mark_x, mark_y = cutout['wcs'].all_world2pix(mark_ra, mark_dec, 0)

            if mark_x is not None and mark_y is not None:
                ax.add_artist(
                    Circle(
                        (mark_x, mark_y),
                        mark_r,
                        edgecolor=mark_color,
                        facecolor='none',
                        ls='-',
                        lw=mark_lw,
                    )
                )

                for _ in [mark_r2, mark_r3]:
                    if _ is not None:
                        ax.add_artist(
                            Circle(
                                (mark_x, mark_y),
                                _,
                                edgecolor=mark_color,
                                facecolor='none',
                                ls='--',
                                lw=mark_lw / 2,
                            )
                        )

            if curplot > nplots:
                break

    if show_title:
        if title is None:
            title = cutout['meta'].get('name', 'unnamed')
            if 'time' in cutout['meta']:
                title += ' at %s' % cutout['meta']['time'].to_value('iso')

            if 'mag_filter_name' in cutout['meta']:
                title += ' : ' + cutout['meta']['mag_filter_name']
                if (
                    'mag_color_name' in cutout['meta']
                    and 'mag_color_term' in cutout['meta']
                    and cutout['meta']['mag_color_term'] is not None
                ):
                    title += ' ' + photometry.format_color_term(
                        cutout['meta']['mag_color_term'],
                        color_name=cutout['meta']['mag_color_name'],
                    )

            if 'mag_limit' in cutout['meta']:
                title += ' : limit %.2f' % cutout['meta']['mag_limit']

            if 'mag_calib' in cutout['meta']:
                title += ' : mag = %.2f $\\pm$ %.2f' % (
                    cutout['meta'].get('mag_calib', np.nan),
                    cutout['meta'].get('mag_calib_err', cutout['meta'].get('magerr', np.nan)),
                )

            if additional_title:
                title += ' : ' + additional_title

        fig.suptitle(title)


def plot_photometric_match(m, ax=None, mode='mag', show_masked=True, show_final=True, **kwargs):
    """Plot photometric match diagnostics.

    Displays various representations of the results returned by
    :func:`stdpipe.photometry.match` or :func:`stdpipe.pipeline.calibrate_photometry`.

    Available modes:

    - ``'mag'`` — photometric residuals vs catalogue magnitude
    - ``'normed'`` — residuals divided by errors vs catalogue magnitude
    - ``'color'`` — residuals vs catalogue color
    - ``'zero'`` — spatial map of the empirical zero point
    - ``'model'`` — spatial map of the zero-point model
    - ``'residuals'`` — spatial map of zero-point fitting residuals
    - ``'dist'`` — spatial map of angular separation (arcsec)

    Parameters
    ----------
    m : dict
        Photometric match results dictionary.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; defaults to the current axes.
    mode : str, optional
        Plotting mode (see above).
    show_masked : bool, optional
        If True, include masked objects in the plot.
    show_final : bool, optional
        If True, highlight objects used in the final fit.
    **kwargs
        Additional keyword arguments passed to :func:`binned_map` where applicable.
    """
    if ax is None:
        ax = plt.gca()

    # Textual representation of the photometric model
    model_str = 'Instr = %s' % m.get('cat_col_mag', 'Cat')

    if (
        'cat_col_mag1' in m.keys()
        and 'cat_col_mag2' in m.keys()
        and 'color_term' in m.keys()
        and m['color_term'] is not None
    ):
        model_str += ' ' + photometry.format_color_term(
            m['color_term'],
            color_name='%s - %s'
            % (
                m['cat_col_mag1'],
                m['cat_col_mag2'],
            ),
        )

    model_str += ' + ZP'

    if mode == 'mag':
        ax.errorbar(
            m['cmag'][m['idx0']],
            (m['zero_model'] - m['zero'])[m['idx0']],
            m['zero_err'][m['idx0']],
            fmt='.',
            alpha=0.3,
        )
        if show_final:
            ax.plot(
                m['cmag'][m['idx']],
                (m['zero_model'] - m['zero'])[m['idx']],
                '.',
                alpha=1.0,
                color='red',
                label='Final fit',
            )
        if show_masked:
            ax.plot(
                m['cmag'][~m['idx0']],
                (m['zero_model'] - m['zero'])[~m['idx0']],
                'x',
                alpha=1.0,
                color='orange',
                label='Masked',
            )

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.legend()
        ax.grid(alpha=0.2)

        ax.set_xlabel(
            'Catalogue %s magnitude' % (m['cat_col_mag'] if 'cat_col_mag' in m.keys() else '')
        )
        ax.set_ylabel('Instrumental - Model')

        ax.set_title(
            '%d of %d unmasked stars used in final fit' % (np.sum(m['idx']), np.sum(m['idx0']))
        )

        ax.text(0.02, 0.05, model_str, transform=ax.transAxes)

    elif mode == 'normed':
        ax.plot(
            m['cmag'][m['idx0']],
            ((m['zero_model'] - m['zero']) / m['zero_err'])[m['idx0']],
            '.',
            alpha=0.3,
        )
        if show_final:
            ax.plot(
                m['cmag'][m['idx']],
                ((m['zero_model'] - m['zero']) / m['zero_err'])[m['idx']],
                '.',
                alpha=1.0,
                color='red',
                label='Final fit',
            )
        if show_masked:
            ax.plot(
                m['cmag'][~m['idx0']],
                ((m['zero_model'] - m['zero']) / m['zero_err'])[~m['idx0']],
                'x',
                alpha=1.0,
                color='orange',
                label='Masked',
            )

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.axhline(-3, ls=':', color='black', alpha=0.3)
        ax.axhline(3, ls=':', color='black', alpha=0.3)
        ax.legend()
        ax.grid(alpha=0.2)

        ax.set_xlabel(
            'Catalogue %s magnitude' % (m['cat_col_mag'] if 'cat_col_mag' in m.keys() else '')
        )
        ax.set_ylabel('(Instrumental - Model) / Error')

        ax.set_title(
            '%d of %d unmasked stars used in final fit' % (np.sum(m['idx']), np.sum(m['idx0']))
        )

        ax.text(0.02, 0.05, model_str, transform=ax.transAxes)

    elif mode == 'color':
        ax.errorbar(
            m['color'][m['idx0']],
            (m['zero_model'] - m['zero'])[m['idx0']],
            m['zero_err'][m['idx0']],
            fmt='.',
            alpha=0.3,
        )
        if show_final:
            ax.plot(
                m['color'][m['idx']],
                (m['zero_model'] - m['zero'])[m['idx']],
                '.',
                alpha=1.0,
                color='red',
                label='Final fit',
            )
        if show_masked:
            ax.plot(
                m['color'][~m['idx0']],
                (m['zero_model'] - m['zero'])[~m['idx0']],
                'x',
                alpha=1.0,
                color='orange',
                label='Masked',
            )

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.legend()
        ax.grid(alpha=0.2)

        ax.set_xlabel(
            'Catalogue %s color'
            % (m['cat_col_mag1'] + '-' + m['cat_col_mag2'] if 'cat_col_mag1' in m.keys() else '')
        )
        ax.set_ylabel('Instrumental - Model')

        ax.set_title('color term = ' + photometry.format_color_term(m['color_term']))

        ax.text(0.02, 0.05, model_str, transform=ax.transAxes)

    elif mode == 'zero':
        if show_final:
            binned_map(
                m['ox'][m['idx']],
                m['oy'][m['idx']],
                m['zero'][m['idx']],
                ax=ax,
                **kwargs,
            )
        else:
            binned_map(
                m['ox'][m['idx0']],
                m['oy'][m['idx0']],
                m['zero'][m['idx0']],
                ax=ax,
                **kwargs,
            )
        ax.set_title('Zero point')

    elif mode == 'model':
        binned_map(
            m['ox'][m['idx0']],
            m['oy'][m['idx0']],
            m['zero_model'][m['idx0']],
            ax=ax,
            **kwargs,
        )
        ax.set_title('Model')

    elif mode == 'residuals':
        binned_map(
            m['ox'][m['idx0']],
            m['oy'][m['idx0']],
            (m['zero_model'] - m['zero'])[m['idx0']],
            ax=ax,
            **kwargs,
        )
        ax.set_title('Instrumental - model')

    elif mode == 'dist':
        binned_map(
            m['ox'][m['idx']],
            m['oy'][m['idx']],
            m['dist'][m['idx']] * 3600,
            ax=ax,
            **kwargs,
        )
        ax.set_title(
            '%d stars: mean displacement %.1f arcsec, median %.1f arcsec'
            % (
                np.sum(m['idx']),
                np.mean(m['dist'][m['idx']] * 3600),
                np.median(m['dist'][m['idx']] * 3600),
            )
        )

    return ax


def plot_detection_limit(
    obj,
    sn=5,
    mag_name=None,
    obj_col_mag='mag_calib',
    obj_col_mag_err='magerr',
    show_local=True,
    ax=None,
):
    """Plot S/N vs magnitude with detection limit model.

    Parameters
    ----------
    obj : astropy.table.Table
        Table with calibrated object detections.
    sn : float, optional
        S/N threshold defining the detection limit.
    mag_name : str, optional
        Axis label for the magnitude.
    obj_col_mag : str, optional
        Column name for calibrated magnitude.
    obj_col_mag_err : str, optional
        Column name for magnitude error.
    show_local : bool, optional
        If True, overlay local per-object detection limits from ``bg_fluxerr``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; defaults to the current axes.
    """
    if ax is None:
        ax = plt.gca()

    mag = obj[obj_col_mag]
    mag_sn = 1 / obj[obj_col_mag_err]

    ax.plot(mag, mag_sn, '.', alpha=(0.2 if len(mag_sn) > 1000 else 0.4), label='Objects')

    ax.axhline(sn, color='black', ls='--', label=f"S/N={sn}")

    mag0, sn_model = photometry.get_detection_limit_sn(mag, mag_sn, sn=sn, get_model=True)
    if mag0 is not None:
        x0 = np.linspace(np.nanmin(mag) - 1, max(np.nanmax(mag), mag0 + 1))
        ax.plot(x0, sn_model(x0), '-', color='red', label='Model')
        ax.axvline(mag0, ls=':', color='red', alpha=0.3, label=f"S/N={sn}")
        ax.set_title(f"S/N={sn} limit is {mag0:.2f}")
    else:
        ax.set_title("Cannot estimate detection limit")

    # Local bg rms detection limit
    if show_local and 'bg_fluxerr' in obj.colnames:
        fluxerr = obj['bg_fluxerr']
        zero = obj['mag_calib'] - obj['mag']
        maglim = -2.5 * np.log10(sn * fluxerr) + zero

        ax.plot(
            maglim,
            np.ones_like(maglim) * sn,
            'o',
            color='orange',
            label='Local RMS',
            alpha=0.2,
        )
        # ax.violinplot(maglim, [sn], vert=False, showmedians=True, color='orange')

    ax.set_yscale('log')

    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_xlabel(mag_name)
    ax.set_ylabel('Signal / Noise')


def plot_mag_histogram(
    obj,
    cat=None,
    cat_col_mag=None,
    sn=None,
    obj_col_mag='mag_calib',
    obj_col_mag_err='magerr',
    accept_flags=0,
    ax=None,
):
    """Plot a histogram of calibrated magnitudes.

    Parameters
    ----------
    obj : astropy.table.Table
        Table with calibrated object detections.
    cat : astropy.table.Table, optional
        Reference catalogue; its magnitudes are overlaid as a separate histogram.
    cat_col_mag : str, optional
        Column name for catalogue magnitudes.
    sn : float, optional
        If set, overplot the S/N detection limit as a vertical line.
    obj_col_mag : str, optional
        Column name for object calibrated magnitude.
    obj_col_mag_err : str, optional
        Column name for object magnitude error.
    accept_flags : int, optional
        Bitmask of acceptable object flags (objects matching this mask are shown
        as unflagged).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; defaults to the current axes.
    """
    if ax is None:
        ax = plt.gca()

    mag = obj[obj_col_mag]
    mag_sn = 1 / obj[obj_col_mag_err]

    idx = (obj['flags'] & ~accept_flags) == 0

    vmin = np.nanmin(mag)
    vmax = np.nanmax(mag)

    if cat and cat_col_mag in cat.colnames:
        cmag = cat[cat_col_mag]
        cmag = cmag[(cmag > -10) & (cmag < 30)]  # To exclude common filler values
        vmin = min(vmin, np.nanmin(cmag))
        vmax = max(vmax, np.nanmax(cmag))
    else:
        cmag = None

    vmin = np.floor(vmin)
    vmax = np.ceil(vmax)

    ax.hist(mag, bins=np.linspace(vmin, vmax, 50), alpha=0.4, color='C0', label="Objects")
    ax.hist(mag, bins=np.linspace(vmin, vmax, 50), alpha=0.8, histtype='step', color='C0')

    ax.hist(
        mag[idx],
        bins=np.linspace(vmin, vmax, 50),
        alpha=0.2,
        color='C2',
        label="Unflagged objects",
    )
    ax.hist(
        mag[idx],
        bins=np.linspace(vmin, vmax, 50),
        alpha=0.6,
        histtype='step',
        color='C2',
    )

    if cmag is not None:
        ax.hist(
            cmag,
            bins=np.linspace(vmin, vmax, 50),
            alpha=0.3,
            color='C1',
            label="Catalogue",
        )
        ax.hist(
            cmag,
            bins=np.linspace(vmin, vmax, 50),
            alpha=0.8,
            histtype='step',
            color='C1',
        )
        ax.set_xlabel(cat_col_mag)
    else:
        ax.set_xlabel('Magnitude')

    if sn:
        mag0 = photometry.get_detection_limit_sn(mag, mag_sn, sn=sn)
        if mag0 is not None:
            ax.axvline(mag0, ls=':', color='red', alpha=0.3, label=f"S/N={sn}")

    ax.grid(alpha=0.2)
    ax.legend()


from contextlib import contextmanager


@contextmanager
def figure_saver(filename=None, show=False, tight_layout=True, **kwargs):
    """Context manager that creates a Figure, saves it, and optionally displays it.

    Example::

        with figure_saver('/tmp/figure.png', show=True, figsize=(10, 6)) as fig:
            ax = fig.add_subplot(111)
            ax.plot(x, y, '.-')

    Parameters
    ----------
    filename : str, optional
        Output file path.  Any format supported by Matplotlib is accepted.
    show : bool, optional
        If True, display the figure in a Jupyter notebook after saving.
    tight_layout : bool, optional
        If True, call ``fig.tight_layout()`` before saving.
    **kwargs
        Additional keyword arguments passed to :class:`matplotlib.pyplot.Figure`.
    """

    fig = plt.Figure(**kwargs)

    try:
        yield fig
    finally:
        if filename:
            if tight_layout:
                fig.tight_layout()
            fig.savefig(filename, bbox_inches='tight')

        if show:
            try:
                from IPython.core.display import display

                # That should display the figure
                display(fig)
            except:
                pass


def plot_outline(x, y, *args, ax=None, **kwargs):
    """
    Plot the convex hull outline of (x, y) points.

    Parameters
    ----------
    x, y : array-like
        Coordinates of points to outline.
    *args, **kwargs :
        Passed through to `matplotlib.axes.Axes.plot` for line styling.
    ax : matplotlib.axes.Axes or None
        Target axes; defaults to current axes when None.

    Notes
    -----
    - Uses `scipy.spatial.ConvexHull` to compute the outline.
    - Requires at least 3 non-collinear points; otherwise ConvexHull may fail.
    - If you pass masked arrays or NaNs, pre-filter to finite values to avoid
      distorting the hull.
    """
    from scipy.spatial import ConvexHull

    points = np.column_stack((np.array(x), np.array(y)))
    if points.shape[0] < 3:
        return

    hull = ConvexHull(points)

    if ax is None:
        ax = plt.gca()

    kw = kwargs.copy()

    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], *args, **kw)
        if 'label' in kw:
            kw.pop('label')


def cornerplot(
    features, scales=None, lines=None, subsets=None, show_all=True, extra=None, fig=None, **kwargs
):
    """
    Plot pairwise feature scatter panels in an upper-left triangular corner layout.

    Parameters
    ----------
    features : list[tuple[array-like, str]]
        List of (values, label) pairs. Each values array is 1D and aligned.
    scales : list[str] or None
        Per-feature axis scale (e.g., 'linear', 'log'); length must match features.
    lines : list[float] or None
        Per-feature reference line positions; length must match features.
    subsets : list[dict] or None
        Optional subsets to overlay. Each dict must include an 'idx' boolean mask,
        plus any matplotlib style kwargs (e.g., label, color, marker).
    extra : callable or None
        If set, this callable will be called for every panel, with axes instance
        and two features (for x and y axes) as arguments
    show_all : bool
        If True, plot the full dataset in each panel excluding subset masks.
    fig : matplotlib.figure.Figure or None
        Figure to plot into; a new figure is created when None.
    **kwargs :
        Additional matplotlib plot kwargs for the main dataset and subsets.

    Notes
    -----
    - Only off-diagonal panels are drawn, resulting in a (N-1)x(N-1) grid.
    - Subsets are excluded from the "all data" layer when `show_all=True`.
    """

    if fig is None:
        fig = plt.figure()

    N = len(features)

    assert scales is None or len(scales) == len(features)
    assert lines is None or len(lines) == len(features)

    if subsets is None:
        subsets = []

    # Reasonable defaults
    kwargs = kwargs.copy()
    # Plot with markers without lines
    kwargs.setdefault('ls', 'none')
    kwargs.setdefault('marker', '.')

    i = 0
    for iy in range(1, N):
        j = 0
        for ix in range(0, N):
            if ix == iy:
                continue

            i += 1
            j += 1

            if j > (N - iy):
                continue

            col_x, col_y = features[ix], features[iy]

            ax = fig.add_subplot(N - 1, N - 1, i)
            ax.grid(alpha=0.2)

            if ix == 0:
                ax.set_ylabel(col_y[1])  # , rotation=105, labelpad=20)
            if iy == N - 1 or ix > iy:
                ax.set_xlabel(col_x[1])  # , rotation=15)

            # Main plotting

            if show_all:
                idx0 = np.isfinite(col_x[0]) & np.isfinite(col_y[0])
                # Exclude user defined subsets
                for sub in subsets:
                    idx0 &= ~sub['idx']

                ax.plot(col_x[0][idx0], col_y[0][idx0], **kwargs)

            for sub in subsets:
                sub = sub.copy()
                idx = sub.pop('idx')

                kw = kwargs.copy()
                kw.update(sub)

                ax.plot(
                    col_x[0][idx],
                    col_y[0][idx],
                    **kw,
                )

            if callable(extra):
                extra(ax, col_x, col_y)

            if scales is not None:
                ax.set_xscale(scales[ix])
                ax.set_yscale(scales[iy])

            if lines is not None:
                if lines[ix] is not None:
                    ax.axvline(lines[ix], ls='--', color='gray')
                if lines[iy] is not None:
                    ax.axhline(lines[iy], ls='--', color='gray')

            if ix > 0:
                ax.set_yticklabels([])
            if iy < N - 1 and ix <= iy:
                ax.set_xticklabels([])

            if len(subsets):
                legend = ax.legend()

                for _ in legend.legend_handles:
                    _.set_alpha(1)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
