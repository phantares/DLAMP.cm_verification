from collections.abc import Sequence

import matplotlib.pyplot as plt

from .utils import DiscreteColorbar, configure_axis


class GridPlotter:
    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 7.5), facecolor="w")

    def plot_pcolor(
        self,
        xaxis,
        yaxis,
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
        cbar_label: str = "Frequency",
        assign_ctick: bool = False,
        ctick_format: str | None = None,
    ) -> None:

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

        mesh = self.ax.pcolormesh(
            xaxis,
            yaxis,
            data,
            cmap=cbar_configurer.cmap,
            norm=cbar_configurer.norm,
        )

        cbar = self.fig.colorbar(mesh, ax=self.ax, label=cbar_label)
        cbar_configurer.configure_colorbar(
            cbar, cbar_log_scale, assign_ctick, ctick_format
        )

    def plot_line(self, xaxis, yaxis, **line_configs) -> None:
        self.ax.plot(xaxis, yaxis, **line_configs)

    def plot_fill(
        self, axis, vmin: float, vmax: float, direction: str = "x", **fill_configs
    ) -> None:
        fill_func = self.ax.fill_between if direction == "y" else self.ax.fill_betweenx
        fill_func(axis, vmin, vmax, **fill_configs)

    def plot_label(
        self,
        x_log_scale: bool = True,
        xlim: Sequence[float] | None = None,
        xlabel: str = "kg/kg",
        xticks: Sequence | None = None,
        xticklabels: Sequence[str] | None = None,
        ylim: Sequence[float] | None = None,
        invert_y: bool = True,
        ylabel: str = "Pressure (hPa)",
        yticks: Sequence | None = None,
        yticklabels: Sequence[str] | None = None,
        plot_legend: bool = True,
        title_configs: Sequence[dict] = [{"label": "Grid"}],
    ) -> None:

        configure_axis(
            self.ax,
            "x",
            log_scale=x_log_scale,
            lim=xlim,
            label=xlabel,
            ticks=xticks,
            ticklabels=xticklabels,
        )

        configure_axis(
            self.ax,
            "y",
            lim=ylim,
            invert=invert_y,
            label=ylabel,
            ticks=yticks,
            ticklabels=yticklabels,
        )

        if plot_legend:
            self.ax.legend()

        [self.ax.set_title(**title_config) for title_config in title_configs]

    def save(self, filename) -> None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(self.fig)


def plot_profile(
    filename,
    axis,
    prediction,
    target,
    xlim: Sequence[float],
    yticks: Sequence,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(11, 5), facecolor="w", sharey=True)

    for v, (var, pred) in enumerate(prediction.items()):
        tar = target[var]
        ax = axes[v]

        ax.fill_betweenx(
            axis,
            pred,
            tar,
            where=(pred > tar),
            color="#0072B2",
            alpha=0.6,
            interpolate=True,
        )
        ax.fill_betweenx(
            axis,
            pred,
            tar,
            where=(pred < tar),
            color="#D55E00",
            alpha=0.6,
            interpolate=True,
        )

        ax.plot(pred, axis, "k-", label="Prediction", linewidth=2)
        ax.plot(tar, axis, "k:", label="Target", linewidth=2)

        ax.set_xscale("log")
        ax.set_xlim(xlim)

        ax.set_title(rf"${var[0].capitalize()}_{var[1]}$")

    axes[2].set_xlabel("kg/kg")
    configure_axis(axes[0], "y", invert=True, label="Pressure (hPa)", ticks=yticks)

    handles, labels = ax.get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1.0))

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
