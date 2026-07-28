from collections.abc import Sequence

import matplotlib.pyplot as plt

from .utils import DiscreteColorbar, configure_axis


def plot_heat_map(
    filename,
    data,
    cmap: str = "viridis",
    vmin: float = 0,
    vmax: float = 1,
    ncolors: int = 10,
    bounds: Sequence[float] | None = None,
    extend: str = "neither",
    nan_color: str | None = None,
    under_color: str | None = None,
    cbar_log_scale: bool = False,
    cbar_label: str | None = None,
    assign_ctick: bool = False,
    ctick_format: str | None = None,
    xlabel: str = "Variables",
    xticks: Sequence | None = None,
    xticklabels: Sequence[str] | None = None,
    ylabel: str = "Pressure (hPa)",
    yticks: Sequence | None = None,
    yticklabels: Sequence[str] | None = None,
    title_configs: Sequence[dict] = [{"label": "Heat Map"}],
) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), facecolor="w")
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

    img = ax.imshow(
        data,
        aspect="auto",
        cmap=cbar_configurer.cmap,
        norm=cbar_configurer.norm,
        origin="lower",
    )

    cbar = fig.colorbar(img, ax=ax, label=cbar_label)
    cbar_configurer.configure_colorbar(cbar, cbar_log_scale, assign_ctick, ctick_format)

    configure_axis(ax, "x", label=xlabel, ticks=xticks, ticklabels=xticklabels)
    configure_axis(ax, "y", label=ylabel, ticks=yticks, ticklabels=yticklabels)

    [ax.set_title(**title_config) for title_config in title_configs]

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
