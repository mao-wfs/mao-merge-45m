# standard library
import re
from pathlib import Path
from struct import Struct
from typing import (
    Any,
    BinaryIO,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)


# third-party packages
import numpy as np
import pandas as pd
import xarray as xr
from tomlkit import parse
from tqdm import tqdm
from dask.diagnostics import ProgressBar


# constants
DIM = "time"
BIG_ENDIAN = ">"
HEADER_REMOVAL = re.compile(r"\x00| ")
HEADER_SECTION = re.compile(r"^(\$+)(.+)$")
HEADER_SIZE = re.compile(r"HeaderSiz\s+=\s+(\d+)")
HEADER_VALUE = re.compile(r"^\s*(\w+)\s*=\s*(.+)$")
CHANNEL_NAME = re.compile("^CH")
TIME_NAME = "Measured time"
JST_HOURS = np.timedelta64(9, "h")


# type hints
Header = Dict[str, Any]
Data = pd.DataFrame


# main features
def convert(
    path_raw_zarr: Union[Sequence[Path], Path],
    path_dist_zarr: Optional[Path] = None,
    length_per_chunk: int = 1000000,
    overwrite: bool = False,
    progress: bool = False,
):
    """Convert a raw Zarr file(s) to a Zarr file for distribution.

    This function will make a one-dimensional accelerometer outputs
    with time metadata derived from the raw Zarr file.

    Notes:
        The timezone of the Zarr file is not JST but UTC.

    Args:
        path_raw_vdif: Path(s) of the raw VDIF file(s).
        path_dist_zarr: Path of the dist. Zarr file (optional).
        length_per_chunk: Length per chunk in the Zarr file.
        overwrite: Whether to overwrite the dist. Zarr file if exists.
        progress: Whether to show a progress bar.

    Returns:
        Path of the Zarr file for distribution.

    Raises:
        FileExistsError: Raised if the dist. Zarr file exists
            and overwriting is not allowed (default).

    """
    # check the existence of the Zarr file
    if isinstance(path_raw_zarr, Path):
        path_raw_zarr = [path_raw_zarr]

    if path_dist_zarr is None:
        path_dist_zarr = path_raw_zarr[0].with_suffix(".dist.zarr")

    if path_dist_zarr.exists() and not overwrite:
        raise FileExistsError(f"{path_dist_zarr} already exists.")

    # open the Zarr file(s) and concatenate them
    ds_raw = xr.concat(map(xr.open_zarr, path_raw_zarr), DIM).sortby(DIM)
    ds_raw = ds_raw.assign_coords(time=ds_raw.time - JST_HOURS)
    ds_raw = ds_raw.chunk(length_per_chunk)

    # write arrays to the Zarr file
    ds_dist = xr.Dataset()

    for key, da in ds_raw.data_vars.items():
        name, unit = key.split()

        if CHANNEL_NAME.search(name) is None:
            continue

        da.attrs.update(
            long_name=name,
            units=unit.strip("()"),
        )
        ds_dist[f"accelerometer_{name.lower()}"] = da

    ds_dist.time.attrs.update(long_name=TIME_NAME)

    if progress:
        with ProgressBar():
            ds_dist.to_zarr(path_dist_zarr, mode="w")
    else:
        ds_dist.to_zarr(path_dist_zarr, mode="w")

    return path_dist_zarr


def to_zarr(
    path_gbd: Path,
    path_zarr: Optional[Path] = None,
    length_per_chunk: int = 1000000,
    encoding: str = "shift-jis",
    overwrite: bool = False,
    progress: bool = False,
):
    """Convert a GBD file to a Zarr file.

    This function focuses on the conversion between formats.
    The Zarr file for distribution with metadata will be
    made by another function (convert).

    Args:
        path_gbd: Path of the GBD file.
        path_zarr: Path of the Zarr file (optional).
        length_per_chunk: Length per chunk in the Zarr file.
        encoding: Encoding of the GBD file.
        overwrite: Whether to overwrite the Zarr file if exists.
        progress: Whether to show a progress bar.

    Returns:
        Path of the Zarr file.

    Raises:
        FileExistsError: Raised if the Zarr file exists
            and overwriting is not allowed (default).

    """
    # check the existence of the Zarr file
    if path_zarr is None:
        path_zarr = path_gbd.with_suffix(".zarr")

    if path_zarr.exists() and not overwrite:
        raise FileExistsError(f"{path_zarr} already exists.")

    # Read the GBD file and write it to the Zarr file
    data = get_data(path_gbd, encoding, progress)
    ds = cast(xr.Dataset, data.to_xarray()).chunk(length_per_chunk)
    ds.to_zarr(path_zarr, mode="w")

    return path_zarr


def get_header(path: Path, encoding: str = "shift-jis") -> Header:
    """Return the header of a GBD file as a dictionary."""
    with path.open("rb") as f:
        header = f.read(get_header_size(path, encoding)).decode(encoding)
        header_rows = HEADER_REMOVAL.sub("", header).split()

    toml_rows = []
    section = ["", "", ""]

    for row in header_rows:
        toml_rows.append(parse_header_row(row, section))

    return parse("\n".join(toml_rows))


def get_data(
    path: Path,
    encoding: str = "shift-jis",
    progress: bool = False,
) -> Data:
    """Return the data of a GBD file as a pandas DataFrame."""
    header = get_header(path, encoding)

    names = get_data_names(header)
    units = get_data_units(header)
    scales = get_data_scales(header)
    length = get_data_length(header)
    index = get_data_index(header)
    read = get_data_reader(header)

    data = np.zeros([length, len(names)])

    with path.open("rb") as f:
        # skip header
        f.read(get_header_size(path, encoding))

        for i in tqdm(range(length), disable=not progress):
            data[i] = read(f)

    data /= scales

    cols = [f"{name} ({unit})" for name, unit in zip(names, units)]
    return pd.DataFrame(data, index, cols)


# helper features (header)
def get_header_size(path: Path, encoding: str = "shift-jis") -> int:
    """Return the byte size of a GBD file (multiple of 2048)."""
    with path.open("rb") as f:
        header = f.read(2048).decode(encoding)

    if not (m := HEADER_SIZE.search(header)):
        raise ValueError("Could not find header size.")

    return int(m.group(1))


def parse_header_row(row: str, section: List[str]) -> str:
    """Parse a row of a header and return a TOML string."""
    if m := HEADER_SECTION.search(row):
        depth, name = len(m.group(1)), str(m.group(2))

        if depth == 1:
            section[:] = [name, "", ""]
        elif depth == 2:
            section[1:] = [name, ""]
        elif depth == 3:
            section[2:] = [name]
        else:
            raise ValueError(row)

        return "[" + ".".join(filter(len, section)) + "]"

    if m := HEADER_VALUE.search(row):
        key, values = str(m.group(1)), str(m.group(2)).split(",")

        values = list(map(to_parsable, values))

        if len(values) == 1:
            values = values[0]

        return f"{key} = {values!r}"

    raise ValueError(row)


def to_parsable(obj: str) -> Union[str, int]:
    """Convert an object to an integer or a string."""
    try:
        return int(obj)
    except ValueError:
        return obj.strip('"')


# helper features (data)
def get_data_names(header: Header) -> List[str]:
    """Return names (labels) of data."""
    return header["Common"]["Data"]["Order"]


def get_data_length(header: Header) -> int:
    """Return the number of samples along time axis."""
    return int(header["Common"]["Data"]["Counts"])


def get_data_index(header: Header) -> pd.DatetimeIndex:
    """Return index of data along time axis."""
    start = "T".join(header["Common"]["Time"]["Start"])
    freq = header["Common"]["Data"]["Sample"]
    periods = get_data_length(header)

    return pd.date_range(start, None, periods, freq, name=DIM)


def get_data_units(header: Header) -> List[Optional[str]]:
    """Return units (V, mV, ...) of data."""
    names = get_data_names(header)
    scales = header["Measure"]["Scale"]
    units = []

    for name in names:
        if name in scales:
            units.append(scales[name][6])
        else:
            units.append(None)

    return units


def get_data_types(header: Header) -> List[Optional[str]]:
    """Return types (voltage, temperature, humidity) of data."""
    names = get_data_names(header)
    types = []

    for name in names:
        if name in (amp := header["Amp"]):
            types.append(amp[name][1])
        else:
            types.append(None)

    return types


def get_data_ranges(header: Header) -> List[Optional[str]]:
    """Return voltage range (units of V or mV) of data."""
    names = get_data_names(header)
    ranges = []

    for name in names:
        if name in (amp := header["Amp"]):
            ranges.append(amp[name][2])
        else:
            ranges.append(None)

    return ranges


def get_data_scales(header: Header) -> List[float]:
    """Return scale (dimensionless) of data."""
    names = get_data_names(header)
    types = get_data_types(header)
    ranges = get_data_ranges(header)
    scales = []

    for name, type_, range_ in zip(names, types, ranges):
        if type_ is None or range_ is None:
            scales.append(1)
            continue

        if type_ == "TEMP":
            scales.append(10)
            continue

        scale = 1

        if "1" in range_:
            scale *= 2
        elif "2" in range_:
            scale *= 1
        elif "4" in range_:
            scale *= 5
        elif "5" in range_:
            scale *= 4
        else:
            raise ValueError(range_)

        if range_ in ("10mV", "20mV"):
            scale *= 1000
        elif range_ in ("50mV", "100mV", "200mV"):
            scale *= 100
        elif range_ in ("500mV", "1V", "2V"):
            scale *= 10
        elif range_ in ("5V", "10V", "20V"):
            scale *= 1
        elif range_ in ("50V", "100V", "200V"):
            scale *= 0.1
        elif range_ in ("500V", "1000V"):
            scale *= 0.01
        else:
            raise ValueError(range_)

        if "mV" in range_:
            scale *= 1
        else:
            scale *= 1000

        scales.append(scale)

    return scales


def get_data_reader(header: Header) -> Callable[[BinaryIO], Tuple]:
    """Make a binary reader function for data."""
    names = get_data_names(header)

    format_ = ""

    for name in names:
        if "CH" in name:
            format_ += "h"
        elif "Pulse" in name:
            format_ += "L"
        elif "Logic" in name:
            format_ += "H"
        elif "Alarm" in name:
            format_ += "H"
        elif "AlOut" in name:
            format_ += "H"
        elif "Status" in name:
            format_ += "H"
        else:
            raise ValueError(name)

    struct = Struct(BIG_ENDIAN + format_)

    def reader(f: BinaryIO) -> Tuple:
        return struct.unpack(f.read(struct.size))

    return reader
