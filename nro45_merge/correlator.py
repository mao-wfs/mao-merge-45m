# standard library
from logging import getLogger
from struct import Struct
from pathlib import Path
from typing import BinaryIO, Callable, Optional, Tuple


# third-party packages
import numpy as np
import zarr
from tqdm import tqdm


# module-level logger
logger = getLogger(__name__)


# constants
UINT = "I"
SHORT = "h"
LITTLE_ENDIAN = "<"
N_ROWS_VDIF_HEAD = 8
N_ROWS_CORR_HEAD = 64
N_ROWS_CORR_DATA = 512
N_BYTES_PER_UNIT = 1312
N_UNITS_PER_SECOND = 6400


# main features
def to_zarr(
    path_vdif: Path,
    path_zarr: Optional[Path] = None,
    overwrite: bool = False,
    progress: bool = False,
) -> None:
    """Convert a VDIF file to a Zarr file losslessly.

    This function focuses on the conversion between formats.
    The final formatted file with metadata and channel binning
    should be created by another function (format_zarr).

    The output Zarr file from the function has three arrays.

    - vdif_head: 4-byte data (uint32) * 8.
    - corr_head: 4-byte data (uint32) * 64.
    - corr_data: 2-byte data (int16) * 512 (256 re/im values).

    Args:
        path_vdif: Path of the VDIF file.
        path_zarr: Path of the Zarr file (optional).
        overwrite: Whether to overwrite the Zarr file if exists.
        progress: Whether to show a progress bar.

    Raises:
        RuntimeError: Raised if the VDIF file is truncated.

    """
    # check the existence of the Zarr file
    if path_zarr is None:
        path_zarr = Path(path_vdif).with_suffix(".zarr")

    if path_zarr.exists() and not overwrite:
        raise FileExistsError(f"{path_zarr} already exists.")

    # check if the VDIF file is truncated.
    n_units, mod = divmod(path_vdif.stat().st_size, N_BYTES_PER_UNIT)

    if mod:
        raise RuntimeError(f"{path_vdif} is truncated.")

    n_seconds, mod = divmod(n_units, N_UNITS_PER_SECOND)

    if mod:
        raise RuntimeError(f"{path_vdif} is truncated.")

    # prepare an empty zarr file
    z = zarr.open(str(path_zarr), mode="w")

    z.empty(
        name="vdif_head",
        shape=(n_units, N_ROWS_VDIF_HEAD),
        chunks=(N_UNITS_PER_SECOND, N_ROWS_VDIF_HEAD),
        dtype=np.uint32,
    )
    z.empty(
        name="corr_head",
        shape=(n_units, N_ROWS_CORR_HEAD),
        chunks=(N_UNITS_PER_SECOND, N_ROWS_CORR_HEAD),
        dtype=np.uint32,
    )
    z.empty(
        name="corr_data",
        shape=(n_units, N_ROWS_CORR_DATA),
        chunks=(N_UNITS_PER_SECOND, N_ROWS_CORR_DATA),
        dtype=np.int16,
    )

    # Read the VDIF file and write it to the Zarr file
    read_vdif_head = make_binary_reader(N_ROWS_VDIF_HEAD, UINT)
    read_corr_head = make_binary_reader(N_ROWS_CORR_HEAD, UINT)
    read_corr_data = make_binary_reader(N_ROWS_CORR_DATA, SHORT)

    with open(path_vdif, "rb") as f:
        for i in tqdm(range(n_seconds), disable=not progress):
            vdif_head = []
            corr_head = []
            corr_data = []

            for _ in range(N_UNITS_PER_SECOND):
                vdif_head.append(read_vdif_head(f))
                corr_head.append(read_corr_head(f))
                corr_data.append(read_corr_data(f))

            index = slice(
                (i + 0) * N_UNITS_PER_SECOND,
                (i + 1) * N_UNITS_PER_SECOND,
            )

            z.vdif_head[index] = vdif_head
            z.corr_head[index] = corr_head
            z.corr_data[index] = corr_data


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

