# standard library
from datetime import datetime
from pathlib import Path
from re import compile


# dependencies
from mao_merge_45m import correlator, merge


# constants
DATA = Path.home() / "Data" / "2022"
DT_FMT_CORR = "%Y%j%H%M%S"
DT_FMT_ISO8601 = "%Y%m%dT%H%M%SZ"
NAME_FMT_CORR = compile(r"^(\S+?)_(\d{13})_5.dist.zarr$")


def to_iso8601(dt: str) -> str:
    """Convert the datetime format used for correlators to ISO 8601."""
    return datetime.strptime(dt, DT_FMT_CORR).strftime(DT_FMT_ISO8601)


def main() -> None:
    for corr in (DATA / "correlator").glob("*.dist.zarr"):
        if (match := NAME_FMT_CORR.search(corr.name)) is None:
            continue

        try:
            print(f"Start merging {corr}:")

            merged = f"mao45_{to_iso8601(match.group(2))}_if12.zarr.zip"
            merge.merge(
                corr,
                DATA / "merged_20231201" / merged,
                path_accelerometer_zarr=DATA / "accelerometer" / "2022-05.zarr",
                path_weather_zarr=DATA / "weather" / "2022-05.zarr",
                path_antenna_zarr=DATA / "antenna" / "2022-05.zarr",
                path_thermometer_zarr=DATA / "thermometer" / "2022-05.zarr",
                path_power_meter_zarr=DATA / "power_meter" / "2022-05.zarr",
                time_offset=1000,
            )

            print("Done")
        except Exception as error:
            print(f"Failed to merge {corr}: {error}")


if __name__ == "__main__":
    main()
