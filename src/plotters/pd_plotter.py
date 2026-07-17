from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


class PDPlotter:
    def __init__(
        self,
        title: str = "Performance diagram",
    ) -> None:

        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 7.5), facecolor="w")
        self.ax.set_box_aspect(1)

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_yticks(np.arange(0, 1.1, 0.2))
        self.ax.set_xticks(np.arange(0, 1.1, 0.2))
        self.ax.set_xlabel("Success Ratio")
        self.ax.set_ylabel("Probability of Detection")

        self._plot_background()

        self.ax.set_title(title)

    def plot(
        self,
        success_ratios,
        probability_detections,
        colors: Sequence[str],
        annots: Sequence[str],
        labels: Sequence[str] | None = None,
        linestyles: Sequence[str] | None = None,
        legend_order: Sequence[int] | None = None,
    ) -> None:

        plot_line = linestyles is not None and len(linestyles) > 0

        for i in range(len(colors)):
            if plot_line:
                self.ax.plot(
                    success_ratios[i],
                    probability_detections[i],
                    color=colors[i],
                    linewidth=2,
                    linestyle=":",
                    marker="o",
                    markersize=10,
                    zorder=999,
                )

            for j in range(len(annots)):
                self.ax.text(
                    success_ratios[i][j],
                    probability_detections[i][j],
                    annots[j],
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=colors[i],
                    fontsize=16,
                )

            if labels:
                self.ax.plot([], [], color=colors[i], label=labels[i], linewidth=3)

        if labels:
            if not legend_order:
                legend_order = range(len(success_ratios))
            self._set_legend(legend_order)

    def save(self, filename) -> None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(self.fig)

    def _plot_background(self) -> None:
        sample_csi = np.arange(0.1, 1, 0.1)
        sample_sr = np.arange(0.1, 1.01, 0.01)

        sample_pod = np.full([len(sample_csi), len(sample_sr)], 0.0)
        for k in range(len(sample_csi)):
            sample_pod[k, :] = 1 / ((1 / sample_csi[k]) - (1 / sample_sr) + 1)

            maxloc = np.argmax(sample_pod[k, :])
            self.ax.plot(
                sample_sr[maxloc:],
                sample_pod[k, maxloc:],
                color="C2",
                linewidth=2,
                alpha=0.4,
            )

            center = -4
            rotation_angle = np.rad2deg(
                np.arctan2(
                    sample_pod[k, center + 1] - sample_pod[k, center - 1],
                    sample_sr[center + 1] - sample_sr[center - 1],
                )
            )
            self.ax.annotate(
                f"{sample_csi[k]: .1f}",
                xy=(
                    sample_sr[center],
                    sample_pod[k, center],
                ),
                color="C2",
                alpha=0.4,
                backgroundcolor="w",
                transform_rotates_text=True,
                rotation=rotation_angle,
                rotation_mode="anchor",
                ha="right",
                va="center",
            )

        sample_bias = np.array([0.2, 0.5, 0.8, 1, 1.25, 2.0, 5.0])
        sample_sr = np.arange(0, 1.1, 0.1)

        sample_pod = np.full([len(sample_bias), len(sample_sr)], 0.0)
        for j in range(len(sample_bias)):
            sample_pod[j, :] = sample_bias[j] * sample_sr

            self.ax.plot(
                sample_sr,
                sample_pod[j, :],
                color="grey" if sample_bias[j] != 1.0 else "black",
                linewidth=2,
                alpha=0.3 if sample_bias[j] != 1.0 else 0.5,
            )

            if sample_bias[j] <= 1.0:
                center = -3
                xy = (sample_sr[center], sample_pod[j, center])
            else:
                center = int(np.argwhere(sample_pod[j, :] <= 1)[-1][0])
                displacement = 0.05
                xy = (
                    sample_sr[center]
                    - displacement
                    / (sample_pod[j, center + 1] - sample_pod[j, center - 1])
                    * (sample_sr[center + 1] - sample_sr[center - 1]),
                    sample_pod[j, center] - displacement,
                )
            rotation_angle = np.rad2deg(
                np.arctan2(
                    sample_pod[j, center + 1] - sample_pod[j, center - 1],
                    sample_sr[center + 1] - sample_sr[center - 1],
                )
            )
            self.ax.annotate(
                f"{sample_bias[j]: .2f}",
                xy=xy,
                color="grey" if sample_bias[j] != 1.0 else "black",
                alpha=0.3 if sample_bias[j] != 1.0 else 0.5,
                backgroundcolor="w",
                transform_rotates_text=True,
                rotation=rotation_angle,
                rotation_mode="anchor",
                ha="right",
                va="center",
            )

    def _set_legend(self, legend_order: Sequence[int]) -> None:
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(
            [handles[i] for i in legend_order],
            [labels[i] for i in legend_order],
            fontsize=5,
            ncol=3,
            frameon=True,
            edgecolor="black",
            mode="expand",
            bbox_to_anchor=(0, -0.3, 1, 0.2),
        )
