from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt

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

def imshow(image, qq=[0.5,97.5], show_colorbar=True, show_axis=True, ax=None, **kwargs):
    """Simple wrapper around pyplot.imshow with histogram-based intensity scaling"""
    if ax is None:
        ax = plt.gca()

    vmin1,vmax1 = np.percentile(image[np.isfinite(image)], qq)
    if not 'vmin' in kwargs:
        kwargs['vmin'] = vmin1
    if not 'vmax' in kwargs:
        kwargs['vmax'] = vmax1

    if not 'interpolation' in kwargs:
        # Rough heuristic to choose interpolation method based on image dimensions
        if image.shape[0] < 300 and image.shape[1] < 300:
            kwargs['interpolation'] = 'nearest'
        else:
            kwargs['interpolation'] = 'bicubic'

    img = ax.imshow(image, **kwargs)
    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_axis_on()
    if show_colorbar:
        colorbar(img, ax=ax)

def binned_map(x, y, value, bins=16, statistic='mean', qq=[0.5, 97.5], show_colorbar=True, show_dots=False, ax=None, **kwargs):
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
        pass
        colorbar(im, ax=ax)

    if show_dots:
        ax.set_autoscale_on(False)
        ax.plot(x, y, 'b.', alpha=0.3)

def plot_cutout(cutout, fig=None):
    curplot = 1
    Nplots = 3

    if fig is None:
        fig = plt.figure(figsize=[Nplots*4, 4+1.0], dpi=75)

    ax = fig.add_subplot(1, Nplots, curplot)
    curplot += 1
    imshow(cutout['image'], [0.5, 99.5], cmap='Blues_r', show_axis=False, show_colorbar=False, ax=ax)
    ax.set_title('Image')

    ax = fig.add_subplot(1, Nplots, curplot)
    curplot += 1
    imshow(cutout['mask'], vmin=0, vmax=1, cmap='Blues_r', show_axis=False, show_colorbar=False, ax=ax)
    ax.set_title('Mask')

    # ax = fig.add_subplot(1, Nplots, curplot)
    # curplot += 1
    # imshow(cutout['mask'], vmin=0, vmax=1, cmap='Blues_r', show_axis=False, show_colorbar=False, ax=ax)
    # ax.set_title('Mask')

    fig.tight_layout()

    title = cutout['name']
    if 'time' in cutout:
        title += ' at %s' % cutout['time'].to_value('iso')
    if 'mag' in cutout:
        title += ': mag = %.2f $\pm$ %.2f' % (cutout['mag_calib'], cutout['magerr'])
    fig.suptitle(title)
