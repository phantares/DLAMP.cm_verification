import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

_STOPS_DBZ = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
_STOPS_HEX = [
    "#00FFFF",
    "#00A3FF",
    "#0048FF",
    "#00FF00",
    "#00C800",
    "#009600",
    "#FFFF00",
    "#FFC800",
    "#FF7800",
    "#FF0000",
    "#C80000",
    "#960000",
    "#FF00FF",
    "#9600FF",
]

DBZ_MIN, DBZ_MAX = 0, 65
CMAP_NAME = "radar_dbz"


def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def build_radar_cmap(n: int = 256) -> LinearSegmentedColormap:
    """Build the continuous radar dBZ colormap object."""
    positions = [(d - DBZ_MIN) / (DBZ_MAX - DBZ_MIN) for d in _STOPS_DBZ]
    colors = [_hex_to_rgb01(c) for c in _STOPS_HEX]
    return LinearSegmentedColormap.from_list(
        CMAP_NAME, list(zip(positions, colors)), N=n
    )


def register(name: str = CMAP_NAME, force: bool = True) -> None:
    """Register the colormap into matplotlib's global registry so it can be
    referenced by name, e.g. plt.colormaps['radar_dbz'] or cmap='radar_dbz'."""
    cmap = build_radar_cmap()
    try:
        mpl.colormaps.register(cmap, name=name, force=force)
    except AttributeError:
        # older matplotlib fallback
        mpl.cm.register_cmap(name=name, cmap=cmap)


# Register automatically on import so DiscreteColorbar(cmap="radar_dbz", ...) works
register()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # sanity check: sampling 66 points (0..65 dBZ) should reproduce the exact hex table
    cmap = plt.colormaps[CMAP_NAME]
    for d in range(0, 66, 5):
        rgba = cmap(d / DBZ_MAX)
        r, g, b = [int(round(c * 255)) for c in rgba[:3]]
        print(d, f"#{r:02X}{g:02X}{b:02X}")
