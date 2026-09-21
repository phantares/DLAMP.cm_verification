from datetime import datetime
from pathlib import Path


def find_data_files(
    data_dir: Path,
    data_source: str = "testing",
    target_time: datetime | None = None,
    initial_time: datetime | None = None,
) -> list[Path]:

    source_dir = data_dir / data_source

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if initial_time is not None:
        files = [source_dir / f"{initial_time:%Y%m%d_%H%M}.h5"]

    elif target_time is None:
        files = sorted(source_dir.glob("*.h5"))

        if not files:
            raise FileNotFoundError(f"No h5 files found in {source_dir}")

        return files

    else:
        files = [source_dir / f"{target_time:%Y%m}.h5"]

    for file in files:
        if file.exists():
            return [file]

    raise FileNotFoundError(
        f"No data file found for\n"
        f"  source       = {data_source}\n"
        f"  target_time  = {target_time}\n"
        f"  initial_time = {initial_time}\n"
        f"Tried:\n" + "\n".join(f"  - {f}" for f in files)
    )
