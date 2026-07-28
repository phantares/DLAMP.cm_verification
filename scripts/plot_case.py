import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import h5py as h5
import numpy as np
import yaml
from calculators import calculate_water_path
from dotenv import dotenv_values
from plotters import plot_map


def main(
    exp: str,
    target_time: datetime,
    extent: Sequence[float] | None,
    mask_threshold: float = 0.5,
) -> None:

    env = dotenv_values(".env")

    fig_dir = Path(env.get("FIG_DIR"), exp, "cases")
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(env.get("DATA_DIR"), exp)
    file = data_dir / f"{target_time.strftime('%Y%m')}.h5"

    with open(data_dir / "config.yaml", "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    use_mask = configs["model"]["system"]["use_mask"]
    data_configs = configs["dataset"]["var"]

    vars = data_configs["target"] + ["dbz"]
    thresholds = data_configs["threshold"]
    thresholds["dbz"] = 0

    sources = ["prediction", "target"]
    plot_vars = ["IWP", "LWP", "Radar"]
    var_to_plot = {
        "qi": "IWP",
        "qs": "IWP",
        "qg": "IWP",
        "qc": "LWP",
        "qr": "LWP",
        "dbz": "Radar",
    }
    datas = {source: {var: 0 for var in plot_vars} for source in sources}

    with h5.File(file, "r") as f:
        times = [datetime.fromisoformat(t.decode("utf-8")) for t in f["time"]]
        target_index = next((i for i, t in enumerate(times) if t == target_time), None)
        print(times[target_index])

        pressure = f["pressure"][:]
        lat = f["latitude"][:]
        lon = f["longitude"][:]

        i0, j0 = 0, 0
        i1, j1 = len(lat) - 1, len(lon) - 1
        if extent is not None:
            dist_sq = (lon - extent[0]) ** 2 + (lat - extent[2]) ** 2
            min_index_flat = np.argmin(dist_sq)
            i0, j0 = np.unravel_index(min_index_flat, lon.shape)
            dist_sq = (lon - extent[1] - 1) ** 2 + (lat - extent[3]) ** 2
            min_index_flat = np.argmin(dist_sq)
            i1, j1 = np.unravel_index(min_index_flat, lon.shape)

        lat = lat[i0 : i1 + 1, j0 : j1 + 1]
        lon = lon[i0 : i1 + 1, j0 : j1 + 1]

        for var in vars:
            print(var)

            for source in sources:
                data = f[f"{source}s"][var][target_index, :, i0 : i1 + 1, j0 : j1 + 1]
                data[data < thresholds[var]] = 0

                if source == "prediction" and var != "dbz" and use_mask:
                    mask = f["predictions"][f"{var}_mask"][
                        target_index, :, i0 : i1 + 1, j0 : j1 + 1
                    ]
                    data[mask <= mask_threshold] = 0

                datas[source][var_to_plot.get(var, var)] += data

    plot_configs = {
        "WP": {"vmin": -3, "cbar_log_scale": True},
        "Radar": {
            "cmap": "radar_dbz",
            "vmin": 1,
            "vmax": 65,
            "ncolors": 64,
            "extend": "neither",
            "under_color": "w",
            "cbar_label": "dBZ",
        },
    }
    for var in plot_vars:
        for source in sources:
            if var == "Radar":
                data = np.amax(datas[source][var], axis=-3)
                plot_var = var
            else:
                data = calculate_water_path(datas[source][var], pressure)
                plot_var = "WP"

            plot_map(
                fig_dir
                / f"{target_time.strftime('%Y%m%d%H')}_{var.lower()}_{source}.png",
                data,
                lat,
                lon,
                extent=extent,
                title_configs=[
                    {"label": f"{exp}\n{var}: {source.capitalize()}", "loc": "left"},
                    {
                        "label": target_time.strftime("%Y-%m-%d %HUTC"),
                        "loc": "right",
                    },
                ],
                **plot_configs[plot_var],
            )


def extent_type(value):
    if value.lower() == "none":
        return None
    return float(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "exp",
        type=str,
        help="Enter experiment name.",
    )
    parser.add_argument(
        "time",
        type=str,
        help="Enter target time in format YYYYmmddHH",
    )
    parser.add_argument(
        "--extent",
        "-e",
        type=extent_type,
        nargs="+",
        default=[119, 123, 21.2, 26.2],
        help="Enter map extent as 4 numbers 'left right bottom top', or 'None' for no extent setting",
    )
    parser.add_argument(
        "--mask",
        type=float,
        default=0.5,
        help="Enter mask threshold for cloud.",
    )
    args = parser.parse_args()

    if len(args.extent) == 1 and args.extent[0] is None:
        args.extent = None
    elif len(args.extent) == 4 and all(v is not None for v in args.extent):
        pass
    else:
        parser.error(
            "--extent must be either 'None' or exactly 4 numbers (left right bottom top)"
        )

    main(
        args.exp,
        datetime.strptime(args.time, "%Y%m%d%H").replace(tzinfo=timezone.utc),
        extent=args.extent,
        mask_threshold=args.mask,
    )
