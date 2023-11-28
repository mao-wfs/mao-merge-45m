# standard library
from dataclasses import dataclass
import datetime
from pathlib import Path
import re
from typing import Literal, Optional, Sequence, Union


# third-party packages
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from xarray_dataclasses import AsDataset, Attr, Data, Dataof


# constants
LOG_COLS = "time", "power_meter"
JST_HOURS = np.timedelta64(9, "h")


# type hints
T = Literal["time"]


@dataclass
class TotalPower:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Total power"
    units: Attr[str] = "dBm"


@dataclass
class PowerMeter(AsDataset):
    """Representation of power meter logs in xarray."""

    power_meter: Dataof[TotalPower] = 0.0
    """The output of the power meter."""


def convert(
    path_log: Union[Sequence[Path], Path],
    path_zarr: Optional[Path] = None,
    *,
    length_per_chunk: int = 1000000,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Convert a raw log file(s) to a formatted Zarr file.

    This function will make a one-dimensional power meter log outputs
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
            skiprows=7,
            usecols=[0, 1],
            names=["time", "total_power"],
            index_col=0,
        )

        # file_pathから何月日の情報を抽出 --> formatを指定してstr型に変換
        time_id = re.search(r"\d{13}", str(path)).group()
        dt_utc = datetime.datetime.strptime(time_id, "%Y%j%H%M%S")
        dt_jst = dt_utc + datetime.timedelta(hours=9)
        date_ymd_str = dt_jst.strftime("%Y-%m-%dT")

        # 2つのstrを足してindexへ
        df_.index = pd.to_datetime(date_ymd_str + df_.index)

        # ミリ秒のタイムスタンプを追加
        df_["num"] = df_.groupby("time").cumcount() + 1
        df_["num_max"] = df_.groupby("time").max()["num"]

        df_.reset_index(inplace=True)
        df_["s"] = (df_["num"] - 1) / df_["num_max"] * 1000
        df_["time"] = df_["time"] + pd.to_timedelta(np.array(df_["s"]), unit="ms")
        df_.set_index(df_["time"], inplace=True)

        # 不要な列を削除
        df_ = df_.drop(columns=["time", "num", "num_max", "s"])
        df = pd.concat([df, df_])

    df.sort_index(inplace=True)
    # write DataFrame(s) to the Zarr file
    ds = PowerMeter.new(df.total_power)
    ds = ds.assign_coords(time=ds.time - JST_HOURS)
    ds = ds.chunk(length_per_chunk)

    if progress:
        with ProgressBar():
            ds.to_zarr(path_zarr, mode="w")
    else:
        ds.to_zarr(path_zarr, mode="w")

    return path_zarr
