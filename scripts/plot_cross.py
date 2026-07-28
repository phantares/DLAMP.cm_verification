import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import h5py as h5
import numpy as np
import yaml
from calculators import calculate_cross_section_points
from constants import STANDARD_LEVEL
from dotenv import dotenv_values
from plotters import GridPlotter
from scipy.interpolate import griddata


def main(
    exp: str,
    target_time: datetime,
    start_point: Sequence[float],
    end_point: Sequence[float],
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
    source_to_title = {"prediction": exp, "target": "RWRF"}

    var_to_plot = {
        "qi": "q",
        "qs": "q",
        "qg": "q",
        "qc": "q",
        "qr": "q",
        "dbz": "radar",
    }
    plot_configs = {
        "q": {
            "cmap": "Blues",
            "vmin": -8,
            "vmax": -2,
            "ncolors": 8,
            "extend": "both",
            "cbar_log_scale": True,
            "cbar_label": "kg/kg",
        },
        "radar": {
            "cmap": "radar_dbz",
            "vmin": 1,
            "vmax": 65,
            "ncolors": 64,
            "extend": "neither",
            "under_color": "w",
            "cbar_label": "dBZ",
        },
    }
    with h5.File(file, "r") as f:
        times = [datetime.fromisoformat(t.decode("utf-8")) for t in f["time"]]
        target_index = next((i for i, t in enumerate(times) if t == target_time), None)
        print(times[target_index])

        pressure = f["pressure"][:]
        lat = f["latitude"][:]
        lon = f["longitude"][:]

        lat_cs, lon_cs = calculate_cross_section_points(start_point, end_point, 2)
        cross = np.column_stack([lon_cs, lat_cs])
        points = np.column_stack([lon.ravel(), lat.ravel()])

        for var in vars:
            print(var)

            for source in sources:
                data = f[f"{source}s"][var][target_index,]
                data[data < thresholds[var]] = 0

                if source == "prediction" and var != "dbz" and use_mask:
                    mask = f["predictions"][f"{var}_mask"][target_index,]
                    data[mask <= mask_threshold] = 0

                cross_data = np.empty((len(pressure), len(lat_cs)))
                for k in range(len(pressure)):
                    cross_data[k, :] = griddata(
                        points,
                        data[k,].ravel(),
                        cross,
                        method="nearest",
                    )

                title = "radar" if var == "dbz" else var
                plotter = GridPlotter()
                xaxis, x_configs = get_cross_xaxis(
                    start_point, end_point, lat_cs, lon_cs
                )
                plotter.plot_pcolor(
                    xaxis,
                    pressure,
                    cross_data,
                    **plot_configs[var_to_plot[var]],
                )
                plotter.plot_label(
                    x_log_scale=False,
                    yticks=STANDARD_LEVEL[:-4] + STANDARD_LEVEL[-3:-2],
                    plot_legend=False,
                    title_configs=[
                        {
                            "label": rf"${var[0].capitalize()}_{var[1]}$: {source_to_title[source]}",
                            "loc": "left",
                        },
                        {
                            "label": target_time.strftime("%Y-%m-%d %HUTC"),
                            "loc": "right",
                        },
                    ],
                    **x_configs,
                )
                plotter.save(
                    fig_dir
                    / f"{target_time.strftime('%Y%m%d%H')}_cross_{title}_{source}_{start_point[0]}N{start_point[1]}E_{end_point[0]}N{end_point[1]}E.png"
                )


def get_cross_xaxis(start_point, end_point, lat_cs, lon_cs, nticks=6):
    if abs(start_point[0] - end_point[0]) < 1e-6:
        return lon_cs, {"xlabal": "Longitude (°E)"}
    elif abs(start_point[1] - end_point[1]) < 1e-6:
        return lat_cs, {"xlabel": "Latitude (°N)"}

    else:
        idx = np.linspace(0, len(lon_cs) - 1, nticks).round().astype(int)
        xticks = lon_cs[idx]
        xticklabels = [f"{lon_cs[i]:.2f}°E\n{lat_cs[i]:.2f}°N" for i in idx]

        return lon_cs, {"xlabel": "", "xticks": xticks, "xticklabels": xticklabels}


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
        "start_point",
        type=float,
        nargs=2,
        help="Enter start point (lat, lon)",
    )
    parser.add_argument(
        "end_point",
        type=float,
        nargs=2,
        help="Enter end point (lat, lon)",
    )
    parser.add_argument(
        "--mask",
        type=float,
        default=0.5,
        help="Enter mask threshold for cloud.",
    )
    args = parser.parse_args()

    main(
        args.exp,
        datetime.strptime(args.time, "%Y%m%d%H").replace(tzinfo=timezone.utc),
        args.start_point,
        args.end_point,
        mask_threshold=args.mask,
    )
