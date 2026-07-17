from collections.abc import Sequence

import matplotlib.pyplot as plt

from .utils import configure_axis


class BarPlotter:
    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 7.5), facecolor="w")

    def plot_bar(self, bins: Sequence[float], data, **bar_configs) -> None:
        self.ax.bar(bins, data, **bar_configs)

    def plot_label(
        self,
        x_log_scale: bool = True,
        xlim: Sequence[float] | None = None,
        xlabel: str = "kg/kg",
        xticks: Sequence = None,
        xticklabels: Sequence[str] | None = None,
        y_log_scale: bool = True,
        ylim: Sequence[float] = [1e-8, 1],
        ylabel: str = "Probability",
        yticks: Sequence | None = None,
        yticklabels: Sequence[str] | None = None,
        title: str = "Distribution",
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
            log_scale=y_log_scale,
            lim=ylim,
            label=ylabel,
            ticks=yticks,
            ticklabels=yticklabels,
        )

        self.ax.legend()
        self.ax.set_title(title)

    def save(self, filename) -> None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(self.fig)
