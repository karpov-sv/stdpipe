import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context
import pytest

from astropy.visualization import ImageNormalize
from astropy.visualization.stretch import HistEqStretch

from stdpipe import plots


def test_imshow_respects_rcparams_origin_lower():
    image = np.arange(9).reshape(3, 3)

    with rc_context({'image.origin': 'lower'}):
        fig, ax = plt.subplots()
        try:
            img = plots.imshow(image, show_colorbar=False, show_axis=False, ax=ax)
            assert img.get_extent() == [-0.5, 2.5, -0.5, 2.5]
        finally:
            plt.close(fig)


def test_imshow_explicit_origin_overrides_rcparams():
    image = np.arange(9).reshape(3, 3)

    with rc_context({'image.origin': 'lower'}):
        fig, ax = plt.subplots()
        try:
            img = plots.imshow(
                image, show_colorbar=False, show_axis=False, ax=ax, origin='upper'
            )
            assert img.get_extent() == [-0.5, 2.5, 2.5, -0.5]
        finally:
            plt.close(fig)


def test_imshow_respects_xlim_ylim_extent():
    image = np.arange(100).reshape(10, 10)

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(
            image, show_colorbar=False, show_axis=False, ax=ax, xlim=(2, 5), ylim=(3, 7)
        )
        assert img.get_extent() == [1.5, 4.5, 6.5, 2.5]
    finally:
        plt.close(fig)


def test_imshow_downscale_preserves_extent():
    image = np.arange(100).reshape(10, 10)

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(
            image, show_colorbar=False, show_axis=False, ax=ax, max_plot_size=5
        )
        assert img.get_extent() == [-0.5, 9.5, 9.5, -0.5]
    finally:
        plt.close(fig)


def test_imshow_does_not_override_explicit_extent():
    image = np.arange(9).reshape(3, 3)
    extent = [1, 2, 3, 4]

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(
            image,
            show_colorbar=False,
            show_axis=False,
            ax=ax,
            extent=extent,
            origin='lower',
        )
        assert img.get_extent() == extent
    finally:
        plt.close(fig)


def test_imshow_show_axis_toggle():
    image = np.arange(9).reshape(3, 3)

    fig, ax = plt.subplots()
    try:
        plots.imshow(image, show_colorbar=False, show_axis=False, ax=ax)
        assert not ax.axison
    finally:
        plt.close(fig)


def test_imshow_show_colorbar_toggle():
    image = np.arange(9).reshape(3, 3)

    fig, ax = plt.subplots()
    try:
        plots.imshow(image, show_colorbar=False, show_axis=False, ax=ax)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_imshow_interpolation_heuristic():
    small = np.arange(100).reshape(10, 10)
    large = np.arange(160000).reshape(400, 400)

    fig, ax = plt.subplots()
    try:
        img_small = plots.imshow(small, show_colorbar=False, show_axis=False, ax=ax)
        assert img_small.get_interpolation() == 'nearest'
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        img_large = plots.imshow(large, show_colorbar=False, show_axis=False, ax=ax)
        assert img_large.get_interpolation() == 'bicubic'
    finally:
        plt.close(fig)


def test_imshow_qq_overrides_vmin_vmax():
    image = np.arange(9).reshape(3, 3)

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(
            image,
            show_colorbar=False,
            show_axis=False,
            ax=ax,
            qq=[0, 100],
            vmin=2,
            vmax=3,
        )
        assert img.get_clim() == (0.0, 8.0)
    finally:
        plt.close(fig)


def test_imshow_histeq_sets_norm():
    image = np.arange(9).reshape(3, 3).astype(float)

    fig, ax = plt.subplots()
    try:
        img = plots.imshow(
            image,
            show_colorbar=False,
            show_axis=False,
            ax=ax,
            stretch='histeq',
            vmin=1,
            vmax=7,
        )
        assert isinstance(img.norm, ImageNormalize)
        assert isinstance(img.norm.stretch, HistEqStretch)
        assert img.norm.vmin == 1
        assert img.norm.vmax == 7
    finally:
        plt.close(fig)


def test_imshow_matches_matplotlib_extent():
    image = np.arange(12).reshape(3, 4)

    with rc_context({'image.origin': 'upper'}):
        fig, ax = plt.subplots()
        try:
            mpl_img = ax.imshow(image)
            mpl_extent = mpl_img.get_extent()
        finally:
            plt.close(fig)


def test_imshow_matches_matplotlib_extent_lower_origin():
    image = np.arange(12).reshape(3, 4)

    with rc_context({'image.origin': 'lower'}):
        fig, ax = plt.subplots()
        try:
            mpl_img = ax.imshow(image)
            mpl_extent = mpl_img.get_extent()
        finally:
            plt.close(fig)

        fig, ax = plt.subplots()
        try:
            std_img = plots.imshow(image, show_colorbar=False, show_axis=False, ax=ax)
            assert std_img.get_extent() == mpl_extent
        finally:
            plt.close(fig)

        fig, ax = plt.subplots()
        try:
            std_img = plots.imshow(image, show_colorbar=False, show_axis=False, ax=ax)
            assert std_img.get_extent() == mpl_extent
        finally:
            plt.close(fig)


def test_adaptive_binned_map_raises_on_no_finite_points():
    x = np.array([np.nan, np.inf])
    y = np.array([0.0, 1.0])
    value = np.array([np.nan, np.inf])

    with pytest.raises(ValueError, match="No finite data points"):
        plots.adaptive_binned_map(x, y, value)


def test_adaptive_binned_map_handles_target_sn_with_zero_err():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    value = np.array([1.0, 2.0, 3.0, 4.0])
    err = np.zeros_like(value)

    fig, ax = plt.subplots()
    try:
        plots.adaptive_binned_map(
            x,
            y,
            value,
            target_sn=5,
            err=err,
            show_colorbar=False,
            show_axis=False,
            ax=ax,
        )
    finally:
        plt.close(fig)
