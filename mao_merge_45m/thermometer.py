# standard library
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Union


# third-party packages
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from xarray_dataclasses import AsDataset, Attr, Data, Dataof


# constants
LOG_COLS = "time", "temperature_1p5m", "sunshine_flag", "temperature_30m"
JST_HOURS = np.timedelta64(9, "h")
DATE_FORMAT = "%Y/%m/%d,%H:%M:%S"


# type hints
T = Literal["time"]


# dataclasses
@dataclass
class Temperature1p5m:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Temperature at 1.5m"
    units: Attr[str] = "deg C"


@dataclass
class Temperature30m:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Temperature at 30m"
    units: Attr[str] = "deg C"


@dataclass
class SunshineFlag:
    data: Data[T, bool] = False
    long_name: Attr[str] = "Sunshine flag"


@dataclass
class Temperature(AsDataset):
    """Representation of thermometer logs in xarray."""

    temperature_1p5m: Dataof[Temperature1p5m] = 0.0
    """Temperature at 1.5 m above the ground surface at the observation building."""

    temperature_30m: Dataof[Temperature30m] = 0.0
    """Temperature at 30 m above the ground surface at the observation building."""

    sunshine_flag: Dataof[SunshineFlag] = 0.0
    """Sunshine flag."""


def convert(
    path_log: Union[Sequence[Path], Path],
    path_zarr: Optional[Path] = None,
    *,
    length_per_chunk: int = 1000000,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Convert a raw log file(s) to a formatted Zarr file.

    This function will make a one-dimensional thermometer log outputs
    with time metadata derived from the raw log file.

    Args:
        path_log: Path(s) of the raw log file(s).
        path_zarr: Path of the formatted Zarr file (optional).
        length_per_chunk: Length per chunk in the Zarr file.
        overwrite: Whether to overwrite the formatted Zarr file if exists.
        progress: Whether to show a progress bar.

    Returns:
        Path of the formatted Zarr file.

    Raises:
        FileExistsError: Raised if the formatted Zarr file exists
            and overwriting is not allowed (default).

    Notes:
        The timezone of the Zarr file is not JST but UTC.

    """
    # check the existence of the Zarr file
    if isinstance(path_log, Path):
        path_log = [path_log]

    if path_zarr is None:
        path_zarr = path_log[0].with_suffix(".zarr")

    if path_zarr.exists() and not overwrite:
        raise FileExistsError(f"{path_zarr} already exists.")

    # read log file(s) and convert them to DataFrame(s)
    df = pd.DataFrame(
        columns=LOG_COLS[1:],
        index=pd.DatetimeIndex([], name=LOG_COLS[0]),
    )

    for path in path_log:
        df_ = pd.read_csv(
            path,
            header=None,
            # index_col=0,
            names=[
                "Date_YMD",
                "Time_HMS",
                "temperature_15",
                "sunshine_flag",
                "temperature_30",
            ],
            usecols=[0, 1, 5, 7, 11],
        )

        # 1行目と2行目を合成して新しい1行を作成
        dt = df_["Date_YMD"].str.cat(df_["Time_HMS"], sep=",")
        # 日付と時刻の列を結合してtimestamp列を作成 --> indexに設定
        index = pd.to_datetime(dt, format=DATE_FORMAT)
        df_.set_index(index, inplace=True)
        # 不要な列を削除
        df_ = df_.drop(columns=["Date_YMD", "Time_HMS"])
        df = pd.concat([df, df_])

    # write DataFrame(s) to the Zarr file
    ds = Temperature.new(df.temperature_15, df.temperature_30, df.sunshine_flag)
    ds = ds.assign_coords(time=ds.time - JST_HOURS)
    ds = ds.chunk(length_per_chunk)

    if progress:
        with ProgressBar():
            ds.to_zarr(path_zarr, mode="w")
    else:
        ds.to_zarr(path_zarr, mode="w")

    return path_zarr
