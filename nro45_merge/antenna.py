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
LOG_COLS = "time", "antenna_azimuth", "antenna_elevation"
JST_HOURS = np.timedelta64(9, "h")
DATE_FORMAT = "%y%m%d%H%M%S"


# type hints
T = Literal["time"]


# dataclasses
@dataclass
class Azimuth:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Azimuth"
    units: Attr[str] = "degree"


@dataclass
class Elevation:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Elevation"
    units: Attr[str] = "degree"


@dataclass
class Antenna(AsDataset):
    """Representation of antenna logs in xarray."""

    antenna_azimuth: Dataof[Azimuth] = 0.0
    """Azimuth of an antenna."""

    antenna_elevation: Dataof[Elevation] = 0.0
    """Elevation of an antenna."""


def convert(
    path_log: Union[Sequence[Path], Path],
    path_zarr: Optional[Path] = None,
    length_per_chunk: int = 1000000,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Convert a raw log file(s) to a formatted Zarr file.

    This function will make a one-dimensional antenna log outputs
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
            header=0,
            index_col=0,
            names=LOG_COLS,
            sep=r"\s+",
            usecols=range(len(LOG_COLS)),
        )

        index = pd.to_datetime(df_.index, format=DATE_FORMAT)
        df_.set_index(index, inplace=True)

        df = df.append(df_)

    # write DataFrame(s) to the Zarr file
    ds = Antenna.new(df.antenna_azimuth, df.antenna_elevation)
    ds = ds.assign_coords(time=ds.time - JST_HOURS)
    ds = ds.chunk(length_per_chunk)

    if progress:
        with ProgressBar():
            ds.to_zarr(path_zarr, mode="w")
    else:
        ds.to_zarr(path_zarr, mode="w")

    return path_zarr
