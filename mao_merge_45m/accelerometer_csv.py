# standard library
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Sequence, Union


# dependencies
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from xarray_dataclasses import AsDataset, Attr, Data, Dataof


# constants
JST_HOURS = np.timedelta64(9, "h")
LOG_DATECOLS = [1, 2, 3]
LOG_ENCODING = "shift-jis"
LOG_TIMEFMT = "%Y/%m/%d %H:%M:%S %f"
LOG_UNITSSTR = "NO.,Date,Time,us,mV,mV,mV,mV,mV,mV,ﾟC,ﾟC"
LOG_UNITSROW = 27


# type hints
Time = Literal["time"]


@dataclass
class CH1:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH1"
    units: Attr[str] = "mV"


@dataclass
class CH2:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH2"
    units: Attr[str] = "mV"


@dataclass
class CH3:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH3"
    units: Attr[str] = "mV"


@dataclass
class CH4:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH4"
    units: Attr[str] = "mV"


@dataclass
class CH5:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH5"
    units: Attr[str] = "mV"


@dataclass
class CH6:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH6"
    units: Attr[str] = "mV"


@dataclass
class CH7:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH7"
    units: Attr[str] = "degC"


@dataclass
class CH8:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH8"
    units: Attr[str] = "degC"


@dataclass
class Accelerometer(AsDataset):
    """Specification of accelerometer logs in xarray."""

    accelerometer_ch1: Dataof[CH1]
    """Data of CH1."""

    accelerometer_ch2: Dataof[CH2]
    """Data of CH2."""

    accelerometer_ch3: Dataof[CH3]
    """Data of CH3."""

    accelerometer_ch4: Dataof[CH4]
    """Data of CH4."""

    accelerometer_ch5: Dataof[CH5]
    """Data of CH5."""

    accelerometer_ch6: Dataof[CH6]
    """Data of CH6."""

    accelerometer_ch7: Dataof[CH7]
    """Data of CH7."""

    accelerometer_ch8: Dataof[CH8]
    """Data of CH8."""


def convert(
    path_log: Union[Sequence[Path], Path],
    path_zarr: Optional[Path] = None,
    *,
    length_per_chunk: int = 1000000,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Convert raw accelerometer log(s) to a formatted Zarr file.

    This function will make a one-dimensional accelerometer outputs
    with time metadata derived from the raw CSV file(s).

    Args:
        path_log: Path(s) of the raw accelerometer CSV file(s).
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
    if isinstance(path_log, Path):
        path_log = [path_log]

    if path_zarr is None:
        path_zarr = path_log[0].with_suffix(".zarr")

    if path_zarr.exists() and not overwrite:
        raise FileExistsError(f"{path_zarr} already exists.")

    df = pd.DataFrame()

    for path in path_log:
        assert_units(path)
        df = pd.concat([df, read_csv(path)])

    ds = Accelerometer.new(
        accelerometer_ch1=df[4],
        accelerometer_ch2=df[5],
        accelerometer_ch3=df[6],
        accelerometer_ch4=df[7],
        accelerometer_ch5=df[8],
        accelerometer_ch6=df[9],
        accelerometer_ch7=df[10],
        accelerometer_ch8=df[11],
    )
    ds = ds.assign_coords(time=ds.time - JST_HOURS)
    ds = ds.chunk(length_per_chunk)

    if progress:
        with ProgressBar():
            ds.to_zarr(path_zarr, mode="w")
    else:
        ds.to_zarr(path_zarr, mode="w")

    return path_zarr


def assert_units(path: Path) -> None:
    """Check if units of an accelerometer log is valid."""
    with open(path, encoding=LOG_ENCODING) as f:
        lines = f.readlines(4096)

    assert lines[LOG_UNITSROW].strip() == LOG_UNITSSTR


def read_csv(path: Path) -> pd.DataFrame:
    """Custom read_csv function dedicated to accelerometer logs."""
    date_parser = partial(pd.to_datetime, format=LOG_TIMEFMT)

    return (
        pd.read_csv(
            path,
            header=None,
            skiprows=LOG_UNITSROW + 1,
            parse_dates=[LOG_DATECOLS],
            index_col="_".join(map(str, LOG_DATECOLS)),
            usecols=range(1, len(LOG_UNITSSTR.split(","))),
            date_parser=date_parser,
            encoding=LOG_ENCODING,
        )
        .astype(float)
        .groupby(level=0)
        .last()
        .resample("10 ms")
        .interpolate()
    )
