from collections.abc import Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

from .utils import DiscreteColorbar


def plot_map(
    filename,
    data,
    lat,
    lon,
    projection=ccrs.PlateCarree(),
    extent: Sequence[float] | None = None,
    cmap: str = "Blues",
    vmin: float = 0,
    vmax: float = 1,
    ncolors: int = 10,
    bounds: Sequence[float] | None = None,
    extend: str = "both",
    nan_color: str | None = None,
    under_color: str | None = None,
    cbar_log_scale: bool = False,
    cbar_label: str = "kg m⁻²",
    assign_ctick: bool = False,
    ctick_format: str | None = None,
    title_configs: Sequence[dict] = [{"label": ""}],
):

    fig, ax = plt.subplots(
        figsize=(10, 7.5),
        subplot_kw={"projection": projection},
    )

    cbar_configurer = DiscreteColorbar(
        cmap,
        vmin,
        vmax,
        ncolors,
        bounds,
        extend,
        nan_color,
        under_color,
        cbar_log_scale,
    )

    im = ax.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cbar_configurer.cmap,
        norm=cbar_configurer.norm,
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="grey", linestyle="--")

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.MultipleLocator(1)
    gl.ylocator = mticker.MultipleLocator(1)

    if extent is None:
        extent = [np.amin(lon), np.amax(lon), np.amin(lat), np.amax(lat)]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        fraction=0.03,
        shrink=0.9,
        label=cbar_label,
    )
    cbar_configurer.configure_colorbar(cbar, cbar_log_scale, assign_ctick, ctick_format)

    [ax.set_title(**title_config) for title_config in title_configs]

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
