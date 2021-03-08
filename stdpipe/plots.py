from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt

from astropy.stats import mad_std

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

def imshow(image, qq=None, show_colorbar=True, show_axis=True, asinh=False, log=False, ax=None, **kwargs):
    """Simple wrapper around pyplot.imshow with histogram-based intensity scaling"""
    if ax is None:
        ax = plt.gca()

    image = image.astype(np.double)

    if asinh:
        data = np.arcsinh(image)
    elif log:
        data = image - np.nanmin(image)
        data = image + 0.1*mad_std(image) if mad_std(image) > 0 else image + 1
        data = np.log10(image)
    else:
        data = image

    if qq is None and 'vmin' not in kwargs and 'vmax' not in kwargs:
        # Sane defaults for quantiles if no manual limits provided
        qq = [0.5, 99.5]

    if qq is not None:
        # Presente of qq quantiles overwrites vmin/vmax even if they are present
        kwargs['vmin'],kwargs['vmax'] = np.percentile(data[np.isfinite(data)], qq)

    if not 'interpolation' in kwargs:
        # Rough heuristic to choose interpolation method based on image dimensions
        if image.shape[0] < 300 and image.shape[1] < 300:
            kwargs['interpolation'] = 'nearest'
        else:
            kwargs['interpolation'] = 'bicubic'

    img = ax.imshow(data, **kwargs)
    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_axis_on()
    if show_colorbar:
        colorbar(img, ax=ax)

def binned_map(x, y, value, bins=16, statistic='mean', qq=[0.5, 97.5], show_colorbar=True, show_axis=True, show_dots=False, ax=None, **kwargs):
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
        ax.plot(x, y, 'b.', alpha=0.3)

def plot_cutout(cutout, fig=None, nplots=3, **kwargs):
    curplot = 1

    if fig is None:
        fig = plt.figure(figsize=[nplots*4, 4+1.0], dpi=75)

    for name in ['image', 'template', 'diff', 'mask']:
        if name in cutout:
            ax = fig.add_subplot(1, nplots, curplot)
            curplot += 1

            params = {'asinh': True if name in ['image', 'template'] else False,
                      'cmap': 'Blues_r',
                      'show_colorbar': False,
                      'show_axis': False}

            # if not kwargs.get('asinh', False) and not kwargs.get('log', False):
            #     median,mad = np.nanmedian(cutout[name]), mad_std(cutout[name], ignore_nan=True)
            #     params['vmin'] = median - 3*mad
            #     params['vmax'] = median + 10*mad
            # elif 'qq' not in kwargs:
            #         kwargs['qq'] = [0, 100]

            params.update(kwargs)

            imshow(cutout[name], ax=ax, **params)
            ax.set_title(name.upper())

            if curplot > nplots:
                break

    fig.tight_layout()

    title = cutout['name']
    if 'time' in cutout:
        title += ' at %s' % cutout['time'].to_value('iso')
    if 'mag' in cutout:
        title += ': mag = %.2f $\pm$ %.2f' % (cutout['mag_calib'], cutout['magerr'])
    fig.suptitle(title)

def plot_photometric_match(m, ax=None, mode='mag', **kwargs):
    if ax is None:
        ax = plt.gca()

    if mode == 'mag':
        ax.errorbar(m['cmag'][m['idx0']], (m['zero']-m['zero_model'])[m['idx0']], m['zero_err'][m['idx0']], fmt='.', alpha=0.3)
        ax.plot(m['cmag'][m['idx']], (m['zero']-m['zero_model'])[m['idx']], '.', alpha=1.0, color='red', label='Final fit')
        ax.plot(m['cmag'][~m['idx0']], (m['zero']-m['zero_model'])[~m['idx0']], 'x', alpha=1.0, color='orange', label='Masked')

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.legend()

        ax.set_xlabel('Catalogue magnitude')
        ax.set_ylabel('Model - Instrumental')

        ax.set_title('%d of %d unmasked stars used in final fit' % (np.sum(m['idx0']), np.sum(m['idx'])))

    elif mode == 'color':
        ax.errorbar(m['color'][m['idx0']], (m['zero']-m['zero_model'])[m['idx0']], m['zero_err'][m['idx0']], fmt='.', alpha=0.3)
        ax.plot(m['color'][m['idx']], (m['zero']-m['zero_model'])[m['idx']], '.', alpha=1.0, color='red', label='Final fit')
        ax.plot(m['color'][~m['idx0']], (m['zero']-m['zero_model'])[~m['idx0']], 'x', alpha=1.0, color='orange', label='Masked')

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.legend()

        ax.set_xlabel('Catalogue color')
        ax.set_ylabel('Model - Instrumental')

        ax.set_title('color term = %.2f' % m['color_term'])

    elif mode == 'dist':
        binned_map(m['ox'][m['idx']], m['oy'][m['idx']], m['dist'][m['idx']]*3600, statistic='mean', ax=ax, **kwargs)
        ax.set_title('%d stars: mean displacement %.1f arcsec, median %.1f arcsec' % (np.sum(m['idx']), np.mean(m['dist'][m['idx']]*3600), np.median(m['dist'][m['idx']]*3600)))

    return ax
