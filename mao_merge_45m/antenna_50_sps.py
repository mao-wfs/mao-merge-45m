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
LOG_COLS = (
    "time",
    "antenna_azimuth",
    "antenna_elevation",
    "subref_X",
    "subref_Y",
    "subref_Z1",
    "subref_Z2",
)
JST_HOURS = np.timedelta64(9, "h")


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
class SubrefX:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref X"
    units: Attr[str] = "mm?"


@dataclass
class SubrefY:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref Y"
    units: Attr[str] = "mm?"


@dataclass
class SubrefZ1:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref Z1"
    units: Attr[str] = "mm?"


@dataclass
class SubrefZ2:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref Z2"
    units: Attr[str] = "mm?"


@dataclass
class Antenna(AsDataset):
    """Representation of antenna logs in xarray."""

    antenna_azimuth: Dataof[Azimuth] = 0.0
    """Azimuth of an antenna."""

    antenna_elevation: Dataof[Elevation] = 0.0
    """Elevation of an antenna."""

    subref_X: Dataof[SubrefX] = 0.0
    """X position of a subref."""

    subref_Y: Dataof[SubrefY] = 0.0
    """Y position of a subref."""

    subref_Z1: Dataof[SubrefZ1] = 0.0
    """Z1 position of a subref."""

    subref_Z2: Dataof[SubrefZ2] = 0.0
    """Z2 position of a subref."""


def get_df(
    path_log: Path,
    index: int,
) -> pd.DataFrame:
    """Helper function."""
    return pd.read_csv(
        path_log,
        sep="\s+",
        header=None,
        skiprows=lambda row: row % 10 != index,
        parse_dates=[[1, 2]],
        index_col="1_2",
        usecols=range(1, 8),
        date_parser=partial(pd.to_datetime, format=LOG_TIMEFMT),
    ).astype(float)


def convert(
    path_log: Union[Sequence[Path], Path],
    path_zarr: Optional[Path] = None,
    length_per_chunk: int = 1000000,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Convert a raw 50 sps log file(s) to a formatted Zarr file.

    This function will make a one-dimensional antenna log outputs
    with time metadata derived from the raw 50 sps log file.

    Args:
        path_log: Path(s) of the raw 50 sps log file(s).
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
        # read data as dataframes
        real_az = get_df(path, 0)
        real_el = get_df(path, 1)
        real_X = get_df(path, 4)
        real_Y = get_df(path, 5)
        real_Z1 = get_df(path, 6)
        real_Z2 = get_df(path, 7)

        # make index
        index = real_az.index
        index = index.append(index[-1:] + pd.Timedelta("80 ms"))
        index = index.to_series().asfreq("20 ms").index

        # reshape data
        real_az = real_az.to_numpy().flatten()
        real_el = real_el.to_numpy().flatten()
        real_X = real_X.to_numpy().flatten()
        real_Y = real_Y.to_numpy().flatten()
        real_Z1 = real_Z1.to_numpy().flatten()
        real_Z2 = real_Z2.to_numpy().flatten()

        df_ = pd.DataFrame(
            data={
                LOG_COLS[1]: real_az,
                LOG_COLS[2]: real_el,
                LOG_COLS[3]: real_X,
                LOG_COLS[4]: real_Y,
                LOG_COLS[5]: real_Z2,
                LOG_COLS[6]: real_Z1,
            },
            index=pd.Index(index, name=LOG_COLS[0]),
        )
        df = pd.concat([df, df_])

    # write DataFrame(s) to the Zarr file
    ds = Antenna.new(
        df.antenna_azimuth,
        df.antenna_elevation,
        df.subref_X,
        df.subref_Y,
        df.subref_Z1,
        df.subref_Z2,
    )
    ds = ds.assign_coords(time=ds.time - JST_HOURS)
    ds = ds.chunk(length_per_chunk)

    if progress:
        with ProgressBar():
            ds.to_zarr(path_zarr, mode="w")
    else:
        ds.to_zarr(path_zarr, mode="w")

    return path_zarr
