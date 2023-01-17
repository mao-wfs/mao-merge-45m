# standard library
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Union
from functools import partial


# third-party packages
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from xarray_dataclasses import AsDataset, Attr, Data, Dataof


# constants
LOG_COLS_REAL = (
    "time",
    "antenna_azimuth",
    "antenna_elevation",
    "collimator_azimuth",
    "collimator_elevation",
    "subref_X",
    "subref_Y",
    "subref_Z1",
    "subref_Z2",
)
LOG_COLS_PROG = (
    "time",
    "prog_antenna_azimuth",
    "prog_antenna_elevation",
    "prog_subref_X",
    "prog_subref_Y",
    "prog_subref_Z1",
    "prog_subref_Z2",
)
JST_HOURS = np.timedelta64(9, "h")
LOG_TIMEFMT = "%y%m%d %H%M%S.%f"

# type hints
T = Literal["time"]


# dataclasses
@dataclass
class AntennaAzimuth:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Antenna azimuth"
    units: Attr[str] = "degree"


@dataclass
class AntennaElevation:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Antenna elevation"
    units: Attr[str] = "degree"


@dataclass
class CollimatorAzimuth:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Collimator azimuth"
    units: Attr[str] = "degree"


@dataclass
class CollimatorElevation:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Collimator elevation"
    units: Attr[str] = "degree"


@dataclass
class SubrefX:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref X"
    units: Attr[str] = "mm"


@dataclass
class SubrefY:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref Y"
    units: Attr[str] = "mm"


@dataclass
class SubrefZ1:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref Z1"
    units: Attr[str] = "mm"


@dataclass
class SubrefZ2:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref Z2"
    units: Attr[str] = "mm"


@dataclass
class ProgAntennaAzimuth:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Antenna azimuth (prog)"
    units: Attr[str] = "degree"


@dataclass
class ProgAntennaElevation:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Antenna elevation (prog)"
    units: Attr[str] = "degree"


@dataclass
class ProgSubrefX:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref X (prog)"
    units: Attr[str] = "mm"


@dataclass
class ProgSubrefY:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref Y (prog)"
    units: Attr[str] = "mm"


@dataclass
class ProgSubrefZ1:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref Z1 (prog)"
    units: Attr[str] = "mm"


@dataclass
class ProgSubrefZ2:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Subref Z2 (prog)"
    units: Attr[str] = "mm"


@dataclass
class Antenna(AsDataset):
    """Representation of antenna logs in xarray."""

    antenna_azimuth: Dataof[AntennaAzimuth] = 0.0
    """Azimuth of the antenna."""

    antenna_elevation: Dataof[AntennaElevation] = 0.0
    """Elevation of the antenna."""

    collimator_azimuth: Dataof[CollimatorAzimuth] = 0.0
    """Azimuth of the collimator."""

    collimator_elevation: Dataof[CollimatorElevation] = 0.0
    """Elevation of the collimator."""

    subref_X: Dataof[SubrefX] = 0.0
    """X position of a subref."""

    subref_Y: Dataof[SubrefY] = 0.0
    """Y position of a subref."""

    subref_Z1: Dataof[SubrefZ1] = 0.0
    """Z1 position of a subref."""

    subref_Z2: Dataof[SubrefZ2] = 0.0
    """Z2 position of a subref."""

    prog_antenna_azimuth: Dataof[ProgAntennaAzimuth] = 0.0
    """Azimuth of the antenna (prog)."""

    prog_antenna_elevation: Dataof[ProgAntennaElevation] = 0.0
    """Elevation of the antenna (prog)."""

    prog_subref_X: Dataof[ProgSubrefX] = 0.0
    """X position of a subref (prog)."""

    prog_subref_Y: Dataof[ProgSubrefY] = 0.0
    """Y position of a subref (prog)."""

    prog_subref_Z1: Dataof[ProgSubrefZ1] = 0.0
    """Z1 position of a subref (prog)."""

    prog_subref_Z2: Dataof[ProgSubrefZ2] = 0.0
    """Z2 position of a subref (prog)."""


def get_df_real(
    path_log: Path,
    index: int,
) -> pd.DataFrame:
    """Helper function to read measured values from a log."""
    return (
        pd.read_csv(
            path_log,
            sep=r"\s+",
            header=None,
            skiprows=lambda row: row % 10 != index,
            parse_dates=[[1, 2]],
            index_col="1_2",
            usecols=range(1, 8),
            date_parser=partial(pd.to_datetime, format=LOG_TIMEFMT),
        )
        .astype(float)
        .groupby(level=0)
        .last()
        .resample("100 ms")
        .interpolate()
    )


def get_df_prog(path_log: Path, index: int = 9) -> pd.DataFrame:
    """Helper function to read programmed values from a log."""
    return (
        pd.read_csv(
            path_log,
            sep=r"\s+",
            header=None,
            skiprows=lambda row: row % 10 != index,
            parse_dates=[[1, 2]],
            index_col="1_2",
            usecols=range(1, 9),
            date_parser=partial(pd.to_datetime, format=LOG_TIMEFMT),
        )
        .astype(float)
        .groupby(level=0)
        .last()
        .resample("100 ms")
        .interpolate()
        .interpolate(method="pad")
    )


def convert(
    path_log: Union[Sequence[Path], Path],
    path_zarr: Optional[Path] = None,
    *,
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
        columns=LOG_COLS_REAL[1:],
        index=pd.DatetimeIndex([], name=LOG_COLS_REAL[0]),
    )

    for path in path_log:
        # step 1: read measured values as dataframes
        ant_az = get_df_real(path, 0)
        ant_el = get_df_real(path, 1)
        col_az = get_df_real(path, 2)
        col_el = get_df_real(path, 3)
        subref_X = get_df_real(path, 4)
        subref_Y = get_df_real(path, 5)
        subref_Z1 = get_df_real(path, 6)
        subref_Z2 = get_df_real(path, 7)

        # make index
        index = ant_az.index
        index = index.append(index[-1:] + pd.Timedelta("80 ms"))
        index = index.to_series().asfreq("20 ms").index

        # reshape data
        ant_az = ant_az.to_numpy().flatten()
        ant_el = ant_el.to_numpy().flatten()
        col_az = col_az.to_numpy().flatten()
        col_el = col_el.to_numpy().flatten()
        subref_X = subref_X.to_numpy().flatten()
        subref_Y = subref_Y.to_numpy().flatten()
        subref_Z1 = subref_Z1.to_numpy().flatten()
        subref_Z2 = subref_Z2.to_numpy().flatten()

        df_real = pd.DataFrame(
            data={
                LOG_COLS_REAL[1]: ant_az,
                LOG_COLS_REAL[2]: ant_el,
                LOG_COLS_REAL[3]: col_az,
                LOG_COLS_REAL[4]: col_el,
                LOG_COLS_REAL[5]: subref_X,
                LOG_COLS_REAL[6]: subref_Y,
                LOG_COLS_REAL[7]: subref_Z1,
                LOG_COLS_REAL[8]: subref_Z2,
            },
            index=pd.Index(index, name=LOG_COLS_REAL[0]),
        )

        # step 2: read programmed values as a dataframe
        prog_all = get_df_prog(path)

        df_prog = pd.DataFrame(
            data={
                LOG_COLS_PROG[1]: prog_all.iloc[:, 0],
                LOG_COLS_PROG[2]: prog_all.iloc[:, 1],
                LOG_COLS_PROG[3]: prog_all.iloc[:, 2],
                LOG_COLS_PROG[4]: prog_all.iloc[:, 3],
                LOG_COLS_PROG[5]: prog_all.iloc[:, 4],
                LOG_COLS_PROG[6]: prog_all.iloc[:, 5],
            },
            index=pd.Index(prog_all.index, name=LOG_COLS_PROG[0]),
        )

        # step 3: merge dataframes
        df_all = pd.concat([df_real, df_prog], axis=1).interpolate(method="pad")
        df = pd.concat([df, df_all])

    # write DataFrame(s) to the Zarr file
    ds = Antenna.new(
        df.antenna_azimuth,
        df.antenna_elevation,
        df.collimator_azimuth,
        df.collimator_elevation,
        df.subref_X,
        df.subref_Y,
        df.subref_Z1,
        df.subref_Z2,
        df.prog_antenna_azimuth,
        df.prog_antenna_elevation,
        df.prog_subref_X,
        df.prog_subref_Y,
        df.prog_subref_Z1,
        df.prog_subref_Z2,
    )
    ds = ds.assign_coords(time=ds.time - JST_HOURS)
    ds = ds.chunk(length_per_chunk)

    if progress:
        with ProgressBar():
            ds.to_zarr(path_zarr, mode="w")
    else:
        ds.to_zarr(path_zarr, mode="w")

    return path_zarr
