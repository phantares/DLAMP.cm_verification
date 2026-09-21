import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import argparse
from datetime import datetime, timezone

import h5py as h5
import numpy as np
import wrf
import yaml
from dotenv import dotenv_values
from utils import find_data_files, interpolate_xy, interpolate_z

VAR = "dbz"


def main(
    input_dir,
    exp: str,
    data_source: str,
    initial_time: datetime,
    mask_threshold: float = 0.5,
) -> None:
    env = dotenv_values(".env")

    data_dir = Path(env.get("DATA_DIR"), exp)
    files = find_data_files(data_dir, data_source, initial_time=initial_time)

    sources = ["prediction", "target"]
    vars = ["qr", "qs", "qg"]

    with open(data_dir / "config.yaml", "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    use_mask = configs["model"]["system"]["use_mask"]
    thresholds = configs["dataset"]["var"]["threshold"]

    for file in files:
        print(file.stem)

        datas = {
            source: {var: 0 for var in vars + [VAR] + ["tk", "qv"]}
            for source in sources
        }

        with h5.File(file, "a") as f:
            time = [datetime.fromisoformat(t.decode("utf-8")) for t in f["time"]]

            for var in vars:
                data = f["predictions"][var][:]
                data[data < thresholds[var]] = 0

                if use_mask:
                    mask = f["predictions"][f"{var}_mask"][:]
                    data[mask <= mask_threshold] = 0

                datas["prediction"][var] = data

            h, w = (
                np.size(datas["prediction"][var], -2),
                np.size(datas["prediction"][var], -1),
            )

            target_file = Path(env.get("INPUT_DIR"), f"{time[0]:%Y%m}.h5")
            with h5.File(target_file, "r") as ftar:
                pressure = ftar["pressure"][:]
                time_in = np.array(
                    [datetime.fromisoformat(t.decode("utf-8")) for t in ftar["time"]]
                )
                time_idx = []
                for t in time:
                    idx = np.where(time_in == t)[0][0]
                    time_idx.append(idx)

                lon = ftar["longitude"][:]
                bottom = (np.size(lon, -2) - h) // 2
                left = (np.size(lon, -1) - w) // 2
                lon = lon[bottom : bottom + h, left : left + w]
                lat = ftar["latitude"][bottom : bottom + h, left : left + w]

                tk = ftar["t"][time_idx, :, bottom : bottom + h, left : left + w]
                datas["target"]["tk"] = tk

                qv = ftar["qv"][time_idx, :, bottom : bottom + h, left : left + w]
                datas["target"]["qv"] = qv

                for var in vars:
                    data = ftar[var][time_idx, :, bottom : bottom + h, left : left + w]
                    data[data < thresholds[var]] = 0
                    datas["target"][var] = data

            if data_source == "testing":
                datas["prediction"]["tk"] = tk
                datas["prediction"]["qv"] = qv
            else:
                with h5.File(input_dir / file.name, "r") as fin:
                    p_in = fin["pressure"][:]
                    lon_in = fin["longitude"][:]
                    lat_in = fin["latitude"][:]

                    tk = fin["t"][:]
                    tk = interpolate_xy(tk, lon_in, lat_in, lon, lat)
                    datas["prediction"]["tk"] = interpolate_z(tk, p_in, pressure)

                    qv = fin["qv"][:]
                    qv = interpolate_xy(qv, lon_in, lat_in, lon, lat)
                    datas["prediction"]["qv"] = interpolate_z(qv, p_in, pressure)

            pressure_4d = np.tile(
                pressure[None, :, None, None],
                (len(time_idx), 1, h, w),
            )

            for source in sources:
                datas[source][VAR] = wrf.dbz(
                    pressure_4d * 100,
                    datas[source]["tk"],
                    datas[source]["qv"],
                    datas[source]["qr"],
                    datas[source]["qs"],
                    datas[source]["qg"],
                    use_varint=False,
                    use_liqskin=False,
                    meta=False,
                )

            ref_var = f["predictions"]["qr"]
            dims = [ref_var.dims[i][0] for i in range(len(ref_var.dims))]

            for source in sources:
                group = f[f"{source}s"]
                if VAR in group:
                    del group[VAR]

                ds = group.create_dataset(
                    VAR, data=datas[source][VAR], compression="gzip"
                )
                for d, dim in enumerate(dims):
                    ds.dims[d].attach_scale(dim)


if __name__ == "__main__":
    env = dotenv_values(".env")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "exp",
        type=str,
        help="Enter experiment name.",
    )
    parser.add_argument(
        "--input_dir",
        "-id",
        type=str,
        default=env.get("INPUT_DIR"),
        help="Enter input data dir path.",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="testing",
        help="Enter data source name.",
    )
    parser.add_argument(
        "--initial_time",
        "-i",
        type=str,
        help="Enter initial time of prediction in format YYYYmmddHH.",
    )
    parser.add_argument(
        "--mask",
        type=float,
        default=0.5,
        help="Enter mask threshold for cloud.",
    )
    args = parser.parse_args()

    initial_time = (
        datetime.strptime(args.initial_time, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        if args.initial_time
        else None
    )

    main(
        Path(args.input_dir),
        args.exp,
        args.source,
        initial_time,
        mask_threshold=args.mask,
    )
