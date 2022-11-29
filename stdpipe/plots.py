from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy.stats import mad_std
from astropy.visualization import simple_norm

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import binned_statistic_2d

def colorbar(obj=None, ax=None, size="5%", pad=0.1):
    should_restore = False

    if obj is not None:
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

def imshow(image, qq=None, show_colorbar=True, show_axis=True, stretch='linear', ax=None, **kwargs):
    """Simple wrapper around pyplot.imshow with percentile-based intensity scaling, optional colorbar, etc.

    :param image: Numpy 2d array to display
    :param qq: two-element tuple (or list) with quantiles that define lower and upper limits for image intensity normalization. Default is `[0.5, 99.5]`. Will be superseded by manually provided `vmin` and `vmax` arguments.
    :param show_colorbar: Whether to show a colorbar alongside the image
    :param show_axis: Whether to show the axes around the image
    :param stretch: Image intensity stretching mode - e.g. `linear`, `log`, `asinh`, or anything else supported by Astropy visualization layer
    :param ax: Matplotlib Axes object to be used for plotting, optional
    :param \**kwargs: The rest of parameters will be directly passed to :func:`matplotlib.pyplot.imshow`

    """
    if ax is None:
        ax = plt.gca()

    image = image.astype(np.double)
    good_idx = np.isfinite(image)

    if np.sum(good_idx):
        if qq is None and 'vmin' not in kwargs and 'vmax' not in kwargs:
            # Sane defaults for quantiles if no manual limits provided
            qq = [0.5, 99.5]

        if qq is not None:
            # Presente of qq quantiles overwrites vmin/vmax even if they are present
            kwargs['vmin'],kwargs['vmax'] = np.percentile(image[good_idx], qq)

        if not 'interpolation' in kwargs:
            # Rough heuristic to choose interpolation method based on image dimensions
            if image.shape[0] < 300 and image.shape[1] < 300:
                kwargs['interpolation'] = 'nearest'
            else:
                kwargs['interpolation'] = 'bicubic'

        if stretch and stretch != 'linear':
            kwargs['norm'] = simple_norm(image, stretch, min_cut=kwargs.pop('vmin', None), max_cut=kwargs.pop('vmax', None))

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

def binned_map(x, y, value, bins=16, statistic='mean', qq=[0.5, 97.5], color=None, show_colorbar=True, show_axis=True, show_dots=False, ax=None, **kwargs):
    """Plots various statistical estimators binned onto regular grid from the set of irregular data points (`x`, `y`, `value`).

    :param x: Abscissae of the data points
    :param y: Ordinates of the data points
    :param value: Values of the data points
    :param bins: Number of bins per axis
    :param statistic: Statistical estimator to plot, may be `mean`, `median`, or a function
    :param qq: two-element tuple (or list) with quantiles that define lower and upper limits for image intensity normalization. Default is `[0.5, 97.5]`. Will be superseded by manually provided `vmin` and `vmax` arguments.
    :param color: Color to use for plotting the positions of data points, optional
    :param show_colorbar: Whether to show a colorbar alongside the image
    :param show_axis: Whether to show the axes around the image
    :param show_dots: Whether to overlay the positions of data points onto the plot
    :param ax: Matplotlib Axes object to be used for plotting, optional
    :param \**kwargs: The rest of parameters will be directly passed to :func:`matplotlib.pyplot.imshow`
    :returns: None

    """
    gmag0, xe, ye, binnumbers = binned_statistic_2d(x, y, value, bins=bins, statistic=statistic)

    vmin1,vmax1 = np.percentile(gmag0[np.isfinite(gmag0)], qq)
    if not 'vmin' in kwargs:
        kwargs['vmin'] = vmin1
    if not 'vmax' in kwargs:
        kwargs['vmax'] = vmax1

    if ax is None:
        ax = plt.gca()

    if not 'aspect' in kwargs:
        kwargs['aspect'] = 'auto'

    im = ax.imshow(gmag0.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation='nearest', **kwargs)
    if show_colorbar:
        colorbar(im, ax=ax)

    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_axis_on()

    if show_dots:
        ax.set_autoscale_on(False)
        ax.plot(x, y, '.', color=color, alpha=0.3)

def plot_cutout(cutout, planes=['image', 'template', 'diff', 'mask'], fig=None, mark_x=None, mark_y=None, mark_r=5.0, title=None, additional_title=None, **kwargs):
    """Routine for displaying various image planes from the cutout structure returned by :func:`stdpipe.cutouts.get_cutout`.

    The cutout planes are displayed in a single row, in the order defined by `planes` paremeters. Optionally, circular mark may be overlayed over the planes at the specified pixel position inside the cutout.

    :param cutout: Cutout structure as returned by :func:`stdpipe.cutouts.get_cutout`
    :param planes: List of names of cutout planes to show
    :param fig: Matplotlib figure where to plot, optional
    :param mark_x: `x` coordinate of the overlay mark in cutout coordinates, optional
    :param mark_y: `y` coordinate of the overlay mark in cutout coordinates, optional
    :param mark_r: Radius of the overlay mark in cutout coordinates in pixels, optional
    :param title: The title to show above the cutouts, optional. If not provided, the title will be constructed from various pieces of cutout metadata, plus the contents of `additoonal_title` field, if provided
    :param additional_title: Additional text to append to automatically generated title of the cutout figure.
    :param \**kwargs: All additional parameters will be directly passed to :func:`stdpipe.plots.imshow` calls on individual images

    """

    curplot = 1

    nplots = len(planes)

    if fig is None:
        fig = plt.figure(figsize=[nplots*4, 4+1.0], dpi=75, tight_layout=True)

    for name in planes:
        if name in cutout and cutout[name] is not None:
            ax = fig.add_subplot(1, nplots, curplot)
            curplot += 1

            params = {'stretch': 'asinh' if name in ['image', 'template', 'convolved'] else 'linear',
                      # 'qq': [0.5, 100] if name in ['image', 'template', 'convolved'] else [0.5, 99.5],
                      'cmap': 'Blues_r',
                      'show_colorbar': False,
                      'show_axis': False}

            params.update(kwargs)

            imshow(cutout[name], ax=ax, **params)
            ax.set_title(name.upper())

            if mark_x is not None and mark_y is not None:
                ax.add_artist(Circle((mark_x, mark_y), mark_r, edgecolor='red', facecolor='none', ls='-', lw=2))

            if curplot > nplots:
                break

    if title is None:
        title = cutout['meta'].get('name', 'unnamed')
        if 'time' in cutout['meta']:
            title += ' at %s' % cutout['meta']['time'].to_value('iso')

        if 'mag_filter_name' in cutout['meta']:
                title += ' : ' + cutout['meta']['mag_filter_name']
                if 'mag_color_name' in cutout['meta'] and 'mag_color_term' in cutout['meta'] and cutout['meta']['mag_color_term'] is not None:
                    sign = '-' if cutout['meta']['mag_color_term'] > 0 else '+'
                    title += ' %s %.2f (%s)' % (sign, np.abs(cutout['meta']['mag_color_term']), cutout['meta']['mag_color_name'])

        if 'mag_limit' in cutout['meta']:
            title += ' : limit %.2f' % cutout['meta']['mag_limit']

        if 'mag_calib' in cutout['meta']:
            title += ' : mag = %.2f $\pm$ %.2f' % (cutout['meta'].get('mag_calib', np.nan), cutout['meta'].get('mag_calib_err', cutout['meta'].get('magerr', np.nan)))

        if additional_title:
            title += ' : ' + additional_title

    fig.suptitle(title)

def plot_photometric_match(m, ax=None, mode='mag', show_masked=True, show_final=True, **kwargs):
    """Convenience plotting routine for photometric match results.

    It plots various representations of the photometric match results returned by :func:`stdpipe.photometry.match` or :func:`stdpipe.pipeline.calibrate_photometry`, depending on the `mode` parameter:

    -  `mag` - displays photometric residuals as a function of catalogue magnitude
    -  `normed` - displays normalized (i.e. divided by errors) photometric residuals as a function of catalogue magnitude
    -  `color` - displays photometric residuals as a function of catalogue color
    -  `zero` - displays the map of empirical zero point, i.e. difference of catalogue and instrumental magnitudes for all matched objects
    -  `model` - displays the map of zero point model
    -  `residuals` - displays fitting residuals between zero point and its model
    -  `dist` - displays the map of angular separation between matched objects and stars, in arcseconds

    The parameter `show_dots` controls whether to overlay the positions of the matched objects onto the maps, when applicable.

    :param m: Dictionary with photometric match results
    :param ax: Matplotlib Axes object to be used for plotting, optional
    :param mode: plotting mode - one of `mag`, `color`, `zero`, `model`, `residuals`, or `dist`
    :param show_masked: Whether to show masked objects
    :param show_final: Whether to additionally highlight the objects used for the final fit, i.e. not rejected during iterative thresholding
    :param \**kwargs: the rest of parameters will be directly passed to :func:`stdpipe.plots.binned_map` when applicable.
    :returns: None

    """
    if ax is None:
        ax = plt.gca()

    # Textual representation of the photometric model
    model_str = 'Instr = %s' % m.get('cat_col_mag', 'Cat')

    if 'cat_col_mag1' in m.keys() and 'cat_col_mag2' in m.keys() and 'color_term' in m.keys() and m['color_term'] is not None:
        sign = '-' if m['color_term'] > 0 else '+'
        model_str += ' %s %.2f (%s - %s)' % (sign, np.abs(m['color_term']), m['cat_col_mag1'], m['cat_col_mag2'])

    model_str += ' + ZP'

    if mode == 'mag':
        ax.errorbar(m['cmag'][m['idx0']], (m['zero_model']-m['zero'])[m['idx0']], m['zero_err'][m['idx0']], fmt='.', alpha=0.3)
        if show_final:
            ax.plot(m['cmag'][m['idx']], (m['zero_model']-m['zero'])[m['idx']], '.', alpha=1.0, color='red', label='Final fit')
        if show_masked:
            ax.plot(m['cmag'][~m['idx0']], (m['zero_model']-m['zero'])[~m['idx0']], 'x', alpha=1.0, color='orange', label='Masked')

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.legend()

        ax.set_xlabel('Catalogue %s magnitude' % (m['cat_col_mag'] if 'cat_col_mag' in m.keys() else ''))
        ax.set_ylabel('Instrumental - Model')

        ax.set_title('%d of %d unmasked stars used in final fit' % (np.sum(m['idx']), np.sum(m['idx0'])))

        ax.text(0.02, 0.05, model_str, transform=ax.transAxes)

    elif mode == 'normed':
        ax.plot(m['cmag'][m['idx0']], ((m['zero_model']-m['zero'])/m['zero_err'])[m['idx0']], '.', alpha=0.3)
        if show_final:
            ax.plot(m['cmag'][m['idx']], ((m['zero_model']-m['zero'])/m['zero_err'])[m['idx']], '.', alpha=1.0, color='red', label='Final fit')
        if show_masked:
            ax.plot(m['cmag'][~m['idx0']], ((m['zero_model']-m['zero'])/m['zero_err'])[~m['idx0']], 'x', alpha=1.0, color='orange', label='Masked')

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.axhline(-3, ls=':', color='black', alpha=0.3)
        ax.axhline(3, ls=':', color='black', alpha=0.3)
        ax.legend()

        ax.set_xlabel('Catalogue %s magnitude' % (m['cat_col_mag'] if 'cat_col_mag' in m.keys() else ''))
        ax.set_ylabel('(Instrumental - Model) / Error')

        ax.set_title('%d of %d unmasked stars used in final fit' % (np.sum(m['idx']), np.sum(m['idx0'])))

        ax.text(0.02, 0.05, model_str, transform=ax.transAxes)

    elif mode == 'color':
        ax.errorbar(m['color'][m['idx0']], (m['zero_model']-m['zero'])[m['idx0']], m['zero_err'][m['idx0']], fmt='.', alpha=0.3)
        if show_final:
            ax.plot(m['color'][m['idx']], (m['zero_model']-m['zero'])[m['idx']], '.', alpha=1.0, color='red', label='Final fit')
        if show_masked:
            ax.plot(m['color'][~m['idx0']], (m['zero_model']-m['zero'])[~m['idx0']], 'x', alpha=1.0, color='orange', label='Masked')

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.legend()

        ax.set_xlabel('Catalogue %s color' % (m['cat_col_mag1'] + '-' + m['cat_col_mag2'] if 'cat_col_mag1' in m.keys() else ''))
        ax.set_ylabel('Instrumental - Model')

        ax.set_title('color term = %.2f' % (m['color_term'] or 0.0))

        ax.text(0.02, 0.05, model_str, transform=ax.transAxes)

    elif mode == 'zero':
        if show_final:
            binned_map(m['ox'][m['idx']], m['oy'][m['idx']], m['zero'][m['idx']], ax=ax, **kwargs)
        else:
            binned_map(m['ox'][m['idx0']], m['oy'][m['idx0']], m['zero'][m['idx0']], ax=ax, **kwargs)
        ax.set_title('Zero point')

    elif mode == 'model':
        binned_map(m['ox'][m['idx0']], m['oy'][m['idx0']], m['zero_model'][m['idx0']], ax=ax, **kwargs)
        ax.set_title('Model')

    elif mode == 'residuals':
        binned_map(m['ox'][m['idx0']], m['oy'][m['idx0']], (m['zero_model']-m['zero'])[m['idx0']], ax=ax, **kwargs)
        ax.set_title('Instrumental - model')

    elif mode == 'dist':
        binned_map(m['ox'][m['idx']], m['oy'][m['idx']], m['dist'][m['idx']]*3600, ax=ax, **kwargs)
        ax.set_title('%d stars: mean displacement %.1f arcsec, median %.1f arcsec' % (np.sum(m['idx']), np.mean(m['dist'][m['idx']]*3600), np.median(m['dist'][m['idx']]*3600)))

    return ax

from contextlib import contextmanager

@contextmanager
def figure_saver(filename=None, show=False, tight_layout=True, **kwargs):
    """Simple matplotlib Figure() wrapper, implemented as a context manager.
    It stores the figure to specified file, and optionally displays it interactively if run inside Jupyter.

    Intended to be used as:

    .. code-block:: python

        with figure_saver('/tmp/figure.png', show=True, figsize=(10, 6)) as fig:
            ax = fig.add_subplot(111)
            ax.plot(x, y, '.-')

    :param filename: Name of a file where to store the image. May be in any format supported by Matplotlib
    :param show: Whether to also display the figure inside Jupuyter notebook
    :param tight_layout: Whether to call :code:`fig.tight_layout()` on the figure before saving/displaying it
    :param \**kwargs: The rest of parameters will be directly passed to :func:`matplotlib.pyplot.Figure`

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
