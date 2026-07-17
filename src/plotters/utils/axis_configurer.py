from collections.abc import Sequence


def configure_axis(
    ax,
    axis: str,
    log_scale: bool = False,
    lim: Sequence[float] | None = None,
    invert: bool = False,
    label: str | None = None,
    ticks: Sequence | None = None,
    ticklabels: Sequence[str] | None = None,
) -> None:

    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    set_scale = getattr(ax, f"set_{axis}scale")
    set_lim = getattr(ax, f"set_{axis}lim")
    set_invert = getattr(ax, f"invert_{axis}axis")
    set_label = getattr(ax, f"set_{axis}label")
    set_ticks = getattr(ax, f"set_{axis}ticks")
    set_ticklabels = getattr(ax, f"set_{axis}ticklabels")

    if log_scale:
        set_scale("log")

    if lim is not None:
        set_lim(lim)

    if invert:
        set_invert()

    if label is not None:
        set_label(label)

    if ticks is None and ticklabels is not None:
        ticks = range(len(ticklabels))
    if ticks is not None:
        set_ticks(ticks)

    if ticklabels is not None:
        set_ticklabels(ticklabels)
