__all__ = ["convert", "to_zarr"]


# standard library
from logging import getLogger
from pathlib import Path
from struct import Struct
from typing import BinaryIO, Callable, Optional, Tuple


# third-party packages
import zarr
import dask.array as da
import numpy as np
import xarray as xr
from tqdm import tqdm
from dask.diagnostics import ProgressBar


# module-level logger
logger = getLogger(__name__)


# constants
DIMS = "time", "freq"
UINT = "I"
SHORT = "h"
LITTLE_ENDIAN = "<"
N_ROWS_VDIF_HEAD = 8
N_ROWS_CORR_HEAD = 64
N_ROWS_CORR_DATA = 512
N_BYTES_PER_UNIT = 1312
N_UNITS_PER_SCAN = 64
N_UNITS_PER_SECOND = 6400
N_SCANS_PER_SECOND = 100
N_SCANS_PER_MINUTE = 6000
N_CHANS_FOR_FORMAT = 8192
LOWER_FREQ_MHZ = 16384
SLICE_MONTH = slice(24, 30)
SLICE_SECOND = slice(0, 30)
REF_YEAR = np.datetime64("2000", "Y")
UNIT_MONTH = np.timedelta64(6, "M")
UNIT_SECOND = np.timedelta64(1, "s")
UNIT_MILLISECOND = np.timedelta64(10, "ms")
IMAG_UNIT = np.complex64(1j)  # type: ignore
TIME_NAME = "Measured time"
FREQ_NAME = "Measured frequency"
FREQ_UNIT = "GHz"
CORR_NAME = "Correlator output"
CORR_UNIT = "Arbitrary unit"


# main features
def convert(
    path_raw_zarr: Path,
    path_fmt_zarr: Optional[Path] = None,
    *,
    bin_width: int = 8,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Convert a raw Zarr file to a formatted Zarr file.

    This function will make a two-dimensional correlator output
    with time and freq metadata derived from the raw Zarr file.
    It will also average the output in frequency to reduce the size.

    Args:
        path_raw_vdif: Path of the raw VDIF file.
        path_fmt_zarr: Path of the formatted Zarr file (optional).
        bin_width: Bin width for averaging the output in frequency.
        overwrite: Whether to overwrite the formatted Zarr file if exists.
        progress: Whether to show a progress bar.

    Returns:
        Path of the formatted Zarr file.

    Raises:
        FileExistsError: Raised if the formatted Zarr file exists
            and overwriting is not allowed (default).

    """
    # check the existence of the Zarr file
    if path_fmt_zarr is None:
        path_fmt_zarr = path_raw_zarr.with_suffix(".dist.zarr")

    if path_fmt_zarr.exists() and not overwrite:
        raise FileExistsError(f"{path_fmt_zarr} already exists.")

    # open the Zarr file and reshape arrays
    z = zarr.open(str(path_raw_zarr))
    vdif_head = da.from_zarr(z.vdif_head)  # type: ignore
    corr_data = da.from_zarr(z.corr_data)  # type: ignore

    vdif_head = vdif_head.flatten().reshape(
        (
            vdif_head.shape[0] / N_UNITS_PER_SCAN,
            vdif_head.shape[1] * N_UNITS_PER_SCAN,
        )
    )
    corr_data = corr_data.flatten().reshape(
        (
            corr_data.shape[0] / N_UNITS_PER_SCAN,
            corr_data.shape[1] * N_UNITS_PER_SCAN,
        )
    )

    # make complex correlator array
    correlator = corr_data[:, 0::2] + corr_data[:, 1::2] * IMAG_UNIT

    # use only first 8192 frequency channels
    correlator = correlator[:, :N_CHANS_FOR_FORMAT]

    # make time and freq arrays
    scan = np.arange(len(correlator))
    month = slice_binary(vdif_head[:, 1], SLICE_MONTH) * UNIT_MONTH
    second = slice_binary(vdif_head[:, 0], SLICE_SECOND) * UNIT_SECOND
    millisecond = (scan % N_SCANS_PER_SECOND) * UNIT_MILLISECOND

    time = (REF_YEAR + month + second + millisecond).compute()
    freq = 1e-3 * (LOWER_FREQ_MHZ + np.arange(N_CHANS_FOR_FORMAT))

    # bin channels
    correlator = correlator.reshape(
        (
            correlator.shape[0],
            correlator.shape[1] // bin_width,
            bin_width,
        )
    ).mean(-1)

    freq = freq.reshape(
        (
            freq.shape[0] // bin_width,
            bin_width,
        )
    ).mean(-1)

    # write arrays to the Zarr file
    correlator = correlator.rechunk((N_SCANS_PER_MINUTE, None))

    ds = xr.Dataset(
        dict(
            time=(DIMS[0], time),
            freq=(DIMS[1], freq),
            correlator=(DIMS, correlator),
        )
    )

    ds.time.attrs.update(
        long_name=TIME_NAME,
    )
    ds.freq.attrs.update(
        long_name=FREQ_NAME,
        units=FREQ_UNIT,
    )
    ds.correlator.attrs.update(
        long_name=CORR_NAME,
        units=CORR_UNIT,
    )

    if progress:
        with ProgressBar():
            ds.to_zarr(path_fmt_zarr, mode="w")
    else:
        ds.to_zarr(path_fmt_zarr, mode="w")

    return path_fmt_zarr


def to_zarr(
    path_vdif: Path,
    path_zarr: Optional[Path] = None,
    *,
    seconds_per_chunk: int = 60,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Convert a VDIF file to a Zarr file losslessly.

    This function focuses on the conversion between formats.
    The formatted Zarr file with metadata will be
    made by another function (convert).

    The output Zarr file from the function has three arrays.

    - vdif_head: 4-byte data (uint32) * 8.
    - corr_head: 4-byte data (uint32) * 64.
    - corr_data: 2-byte data (int16) * 512 (256 re/im values).

    Args:
        path_vdif: Path of the VDIF file.
        path_zarr: Path of the Zarr file (optional).
        seconds_per_chunk: Time length per chunk in the Zarr file.
        overwrite: Whether to overwrite the Zarr file if exists.
        progress: Whether to show a progress bar.

    Returns:
        Path of the Zarr file.

    Raises:
        RuntimeError: Raised if the VDIF file is truncated.
        FileExistsError: Raised if the Zarr file exists
            and overwriting is not allowed (default).

    """
    # check the existence of the Zarr file
    if path_zarr is None:
        path_zarr = Path(path_vdif).with_suffix(".zarr")

    if path_zarr.exists() and not overwrite:
        raise FileExistsError(f"{path_zarr} already exists.")

    # check if the VDIF file is truncated.
    n_units, mod = divmod(path_vdif.stat().st_size, N_BYTES_PER_UNIT)

    if mod:
        raise RuntimeError(f"{path_vdif} may be truncated.")

    n_units_per_chunk = N_UNITS_PER_SECOND * seconds_per_chunk
    n_chunks, mod = divmod(n_units, n_units_per_chunk)

    if mod:
        raise RuntimeError("Chunk length could not divide total time.")

    # prepare an empty zarr file
    z = zarr.open(str(path_zarr), mode="w")
    z.empty(  # type: ignore
        name="vdif_head",
        shape=(n_units, N_ROWS_VDIF_HEAD),
        chunks=(n_units_per_chunk, N_ROWS_VDIF_HEAD),
        dtype=np.dtype(UINT),  # type: ignore
    )
    z.empty(  # type: ignore
        name="corr_head",
        shape=(n_units, N_ROWS_CORR_HEAD),
        chunks=(n_units_per_chunk, N_ROWS_CORR_HEAD),
        dtype=np.dtype(UINT),  # type: ignore
    )
    z.empty(  # type: ignore
        name="corr_data",
        shape=(n_units, N_ROWS_CORR_DATA),
        chunks=(n_units_per_chunk, N_ROWS_CORR_DATA),
        dtype=np.dtype(SHORT),  # type: ignore
    )

    # Read the VDIF file and write it to the Zarr file
    read_vdif_head = make_binary_reader(N_ROWS_VDIF_HEAD, UINT)
    read_corr_head = make_binary_reader(N_ROWS_CORR_HEAD, UINT)
    read_corr_data = make_binary_reader(N_ROWS_CORR_DATA, SHORT)

    with open(path_vdif, "rb") as f:
        for i in tqdm(range(n_chunks), disable=not progress):
            vdif_head = []
            corr_head = []
            corr_data = []

            for _ in range(n_units_per_chunk):
                vdif_head.append(read_vdif_head(f))
                corr_head.append(read_corr_head(f))
                corr_data.append(read_corr_data(f))

            index = slice(
                (i + 0) * n_units_per_chunk,
                (i + 1) * n_units_per_chunk,
            )

            z.vdif_head[index] = vdif_head  # type: ignore
            z.corr_head[index] = corr_head  # type: ignore
            z.corr_data[index] = corr_data  # type: ignore

    return path_zarr


def to_vdif(
    path_zarr: Path,
    path_vdif: Optional[Path] = None,
    *,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Convert a Zarr file to a VDIF file losslessly.

    Args:
        path_zarr: Path of the Zarr file.
        path_vdif: Path of the VDIF file (optional).
        overwrite: Whether to overwrite the VDIF file if exists.
        progress: Whether to show a progress bar.

    Returns:
        Path of the VDIF file.

    Raises:
        FileExistsError: Raised if the VDIF file exists
            and overwriting is not allowed (default).

    """
    # check the existence of the Zarr file
    if path_vdif is None:
        path_vdif = Path(path_zarr).with_suffix(".vdif")

    if path_vdif.exists() and not overwrite:
        raise FileExistsError(f"{path_vdif} already exists.")

    # Read the Zarr file and write it to the VDIF file
    z = zarr.open(str(path_zarr), mode="r")
    vdif_head = z.vdif_head  # type: ignore
    corr_head = z.corr_head  # type: ignore
    corr_data = z.corr_data  # type: ignore

    arrays = tqdm(
        zip(vdif_head, corr_head, corr_data),
        total=len(vdif_head),
        disable=not progress,
    )

    with path_vdif.open("wb") as f:
        for units in arrays:
            for unit in units:
                f.write(unit.tobytes())

    return path_vdif


# helper features
def make_binary_reader(
    n_repeat: int,
    dtype: str,
) -> Callable[[BinaryIO], Tuple]:
    """Make a binary reader function."""
    struct = Struct(LITTLE_ENDIAN + dtype * n_repeat)

    def reader(f: BinaryIO) -> Tuple:
        return struct.unpack(f.read(struct.size))

    return reader


def slice_binary(val: int, slice_: slice) -> int:
    """Slice an integer with given bit range."""
    start = int(slice_.start)
    stop = int(slice_.stop)
    length = stop - start

    return (val >> start) & ((1 << length) - 1)
