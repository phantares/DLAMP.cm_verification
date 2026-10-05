from datetime import datetime
from pathlib import Path


def find_data_files(
    data_dir: Path,
    target_time: datetime | None = None,
    initial_time: datetime | None = None,
) -> list[Path]:

    if not data_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {data_dir}")

    if initial_time is not None:
        files = [data_dir / f"{initial_time:%Y%m%d_%H%M}.h5"]

    elif target_time is None:
        files = sorted(data_dir.glob("*.h5"))

        if not files:
            raise FileNotFoundError(f"No h5 files found in {data_dir}")

        return files

    else:
        files = [
            data_dir / f"{target_time:%Y%m}.h5",
            data_dir / f"{target_time:%Y%m%d_%H%M}.h5",
        ]

    for file in files:
        if file.is_file():
            return [file]

    raise FileNotFoundError(
        f"No data file found for\n"
        f"  dir          = {data_dir}\n"
        f"  target_time  = {target_time}\n"
        f"  initial_time = {initial_time}\n"
        f"Tried:\n" + "\n".join(f"  - {f}" for f in files)
    )


def get_prediction_title(exp, data_source, initial_time):
    if data_source == "testing":
        prediction_title = exp
        prediction_name = "prediction"
    elif initial_time is not None:
        prediction_title = f"{exp}\n{data_source}: {initial_time:%Y%m%d %H}Z"
        prediction_name = f"{data_source}_{initial_time.strftime('%Y%m%d%H')}"
    else:
        prediction_title = f"{exp}\n{data_source}"
        prediction_name = data_source

    return prediction_title, prediction_name
