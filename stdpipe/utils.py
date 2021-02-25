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

def breakpoint():
    try:
        from IPython.core.debugger import Tracer
        Tracer()()
    except:
        import pdb
        pdb.set_trace()

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

def crop_image(data, x0, y0, r0, header=None):
    x1,x2 = int(np.floor(x0 - r0)), int(np.ceil(x0 + r0))
    y1,y2 = int(np.floor(y0 - r0)), int(np.ceil(y0 + r0))

    src = [min(max(y1, 0), data.shape[0]),
           max(min(y2, data.shape[0]), 0),
           min(max(x1, 0), data.shape[1]),
           max(min(x2, data.shape[1]), 0)]

    dst = [src[0] - y1, src[1] - y1, src[2] - x1, src[3] - x1]

    sub = np.zeros((y2-y1, x2-x1), data.dtype)
    sub.fill(np.nan)
    sub[dst[0]:dst[1], dst[2]:dst[3]] = data[src[0]:src[1], src[2]:src[3]]

    if header is not None:
        subheader = header.copy()

        # Adjust the WCS keywords if present
        if 'CRPIX1' in subheader and 'CRPIX2' in subheader:
            subheader['CRPIX1'] -= x1
            subheader['CRPIX2'] -= y1

        # FIXME: should we use 0-based or 1-based coordinates here?..

        # Crop target inside cutout
        subheader['CROP_X'] = x0 - x1
        subheader['CROP_Y'] = y0 - y1

        # Crop center inside original frame
        subheader['CROP_X0'] = x0
        subheader['CROP_Y0'] = y0
        subheader['CROP_R0'] = r0

        # Crop position inside original frame
        subheader['CROP_X1'] = x1
        subheader['CROP_X2'] = x2
        subheader['CROP_Y1'] = y1
        subheader['CROP_Y2'] = y2

        return sub, subheader
    else:
        return sub
