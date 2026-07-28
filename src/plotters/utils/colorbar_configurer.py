from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from . import radar_dbz_cmap_module  # noqa: F401


class DiscreteColorbar:
    def __init__(
        self,
        cmap: str = "viridis",
        vmin: float = 0,
        vmax: float = 1,
        ncolors: int = 10,
        bounds: Sequence[float] | None = None,
        extend: str = "neither",
        nan_color: str | None = None,
        under_color: str | None = None,
        log_scale: bool = False,
    ) -> None:

        self.bounds, ncolors = self._compute_bounds(
            vmin, vmax, ncolors, bounds, extend, log_scale
        )

        self.norm = BoundaryNorm(self.bounds, ncolors=ncolors, extend=extend)

        self.cmap = ListedColormap(plt.colormaps[cmap](np.linspace(0, 1, ncolors)))
        if nan_color is not None:
            self.cmap.set_bad(color=nan_color)
        if under_color is not None:
            self.cmap.set_under(color=under_color)

    def _compute_bounds(
        self,
        vmin: float,
        vmax: float,
        ncolors: int,
        bounds: Sequence[float] | None,
        extend: str,
        log_scale: bool,
    ) -> tuple[np.ndarray, int]:
        extend_offset = {"neither": 1, "both": -1}.get(extend, 0)

        if bounds is not None:
            bounds = np.asarray(bounds)
            ncolors = len(bounds) - extend_offset

        else:
            bound_fn = np.logspace if log_scale else np.linspace
            bounds = bound_fn(vmin, vmax, ncolors + extend_offset)

        return bounds, ncolors

    def configure_colorbar(
        self,
        cbar,
        log_scale: bool = False,
        assign_ctick: bool = False,
        ctick_format: str | None = None,
    ) -> None:

        if log_scale:
            cbar.ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
            cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
            return

        if assign_ctick:
            formatter = self._resolve_formatter(ctick_format)
            cbar.set_ticks(self.bounds)
            cbar.set_ticklabels([formatter(b) for b in self.bounds])

    def _resolve_formatter(self, ctick_format: str | None) -> Callable[[float], str]:
        fmt = ctick_format or "{:g}"

        return lambda b: fmt.format(b)
