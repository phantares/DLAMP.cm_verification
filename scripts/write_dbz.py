import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import argparse
from datetime import datetime

import h5py as h5
import numpy as np
import wrf
import yaml
from dotenv import dotenv_values

VAR = "dbz"


def main(exp: str, mask_threshold: float = 0.5) -> None:
    env = dotenv_values(".env")

    data_dir = Path(env.get("DATA_DIR"), exp)
    files = sorted(data_dir.glob("*.h5"))
    target_dir = Path(env.get("INPUT_DIR"))

    sources = ["prediction", "target"]
    vars = ["qr", "qs", "qg"]

    with open(data_dir / "config.yaml", "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    use_mask = configs["model"]["system"]["use_mask"]
    thresholds = configs["dataset"]["var"]["threshold"]

    for file in files:
        month = file.stem[-2:]
        print(month)

        datas = {source: {var: 0 for var in vars + [VAR]} for source in sources}

        with h5.File(file, "a") as f:
            time = [datetime.fromisoformat(t.decode("utf-8")) for t in f["time"]]
            pressure = f["pressure"][:]

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

            with h5.File(target_dir / f"{file.stem}.h5", "r") as fin:
                time_in = np.array(
                    [datetime.fromisoformat(t.decode("utf-8")) for t in fin["time"]]
                )
                time_idx = []
                for t in time:
                    idx = np.where(time_in == t)[0][0]
                    time_idx.append(idx)

                tk = fin["t"][time_idx,]
                bottom = (np.size(tk, -2) - h) // 2
                left = (np.size(tk, -1) - w) // 2
                tk = tk[
                    :,
                    :,
                    bottom : bottom + h,
                    left : left + w,
                ]

                qv = fin["qv"][time_idx, :, bottom : bottom + h, left : left + w]
                for var in vars:
                    data = fin[var][time_idx, :, bottom : bottom + h, left : left + w]
                    data[data < thresholds[var]] = 0
                    datas["target"][var] = data

            pressure_4d = np.tile(
                pressure[None, :, None, None],
                (len(time_idx), 1, h, w),
            )

            for source in sources:
                datas[source][VAR] = wrf.dbz(
                    pressure_4d * 100,
                    tk,
                    qv,
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "exp",
        type=str,
        help="Enter experiment name.",
    )
    parser.add_argument(
        "--mask",
        type=float,
        default=0.5,
        help="Enter mask threshold for cloud.",
    )
    args = parser.parse_args()

    main(args.exp, mask_threshold=args.mask)
