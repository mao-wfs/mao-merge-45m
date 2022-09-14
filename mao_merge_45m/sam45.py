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
JST_HOURS = np.timedelta64(9, "h")

LOG_DTYPE = [
    ("time", "U19"),
    ("array", "U2"),
    ("mode", "U4"),
    ("spec", ("f4", 512)),
]
LOG_TIMEFMT = "%Y%m%d%H%M%S.%f"


# type hints
Time = Literal["time"]
Chan = Literal["chan"]


# dataclasses


@dataclass
class Array(AsDataArray):
    time: Coord[Time, np.datetime64]
    data: Data[Tuple[Time, Chan], np.float32]


@dataclass
class A1:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "SAM45 A1"
    units: Attr[str] = "K"


@dataclass
class A2:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "SAM45 A2"
    units: Attr[str] = "K"


@dataclass
class A3:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "SAM45 A3"
    units: Attr[str] = "K"


@dataclass
class A4:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "SAM45 A4"
    units: Attr[str] = "K"


@dataclass
class A5:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "SAM45 A5"
    units: Attr[str] = "K"


@dataclass
class A6:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "SAM45 A6"
    units: Attr[str] = "K"


@dataclass
class A7:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "SAM45 A7"
    units: Attr[str] = "K"


@dataclass
class A8:
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "SAM45 A8"
    units: Attr[str] = "K"


@dataclass
class SAM45(AsDataset):
    """Representation of SAM45 logs in xarray."""

    sam45_A1: Dataof[A1] = 0.0
    """Data of A1."""

    sam45_A2: Dataof[A2] = 0.0
    """Data of A2."""

    sam45_A3: Dataof[A3] = 0.0
    """Data of A3."""

    sam45_A4: Dataof[A4] = 0.0
    """Data of A4."""

    sam45_A5: Dataof[A5] = 0.0
    """Data of A5."""

    sam45_A6: Dataof[A6] = 0.0
    """Data of A6."""

    sam45_A7: Dataof[A7] = 0.0
    """Data of A7."""

    sam45_A8: Dataof[A8] = 0.0
    """Data of A8."""


def convert(
    path_log: Union[Sequence[Path], Path],
    path_zarr: Optional[Path] = None,
    ch_min: int = 0,
    ch_max: int = 4096,
    T_AMB: float = 273.0,
    length_per_chunk: int = 1000000,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Convert a raw SAM45 log file(s) to a formatted Zarr file.

    This function will make a one-dimensional antenna log outputs
    with time metadata derived from the raw SAM45 log file.

    Args:
        path_log: Path(s) of the raw SAM45 log file(s).
        ch_min: Minimum channel used for channel binning.
        ch_max: Maximum channel used for channel binning.
        T_AMB: Ambient temperature for intensity calibration.
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
    dl = []

    for path in path_log:
        # read data as datasets
        data = np.genfromtxt(path, dtype=LOG_DTYPE)
        ds_ = xr.Dataset()

        for array in np.unique(data["array"]):
            where = data["array"] == array
            spec = data["spec"][where]
            mode = data["mode"][where]
            time = data["time"][where]

            calibrated = (
                T_AMB
                * (spec[mode == "ON"] - spec[mode == "SKY"])
                / (spec[mode == "R"] - spec[mode == "SKY"])
            )
            datetime = pd.to_datetime(time[mode == "ON"], format=LOG_TIMEFMT)
            ds_[array] = Array.new(datetime.to_numpy(), calibrated)

        dl.append(ds_.sel(chan=slice(ch_min, ch_max)).mean("chan"))

    # write DataFrame(s) to the Zarr file
    ds = xr.concat(dl, dim="time")
    ds = SAM45.new(
        ds["A1"].values,
        ds["A2"].values,
        ds["A3"].values,
        ds["A4"].values,
        ds["A5"].values,
        ds["A6"].values,
        ds["A7"].values,
        ds["A8"].values,
    )
    ds = ds.assign_coords(time=ds.time - JST_HOURS)
    ds = ds.chunk(length_per_chunk)

    if progress:
        with ProgressBar():
            ds.to_zarr(path_zarr, mode="w")
    else:
        ds.to_zarr(path_zarr, mode="w")

    return path_zarr
