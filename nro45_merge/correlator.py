# standard library
from logging import getLogger
from struct import Struct
from pathlib import Path
from typing import BinaryIO, Callable, Tuple


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


# main features
def to_zarr(vdif: Path, progress: bool = True) -> Path:
    """Convert a VDIF file to a Zarr dataset.

    Args:
        vdif: Path of the VDIF file.
        progress: Whether to show a progress bar.

    Returns:
        Path of the Zarr dataset.

    Raises:
        RuntimeError: Raised if the VDIF file is broken.

    """
    # check if the VDIF file is broken
    n_units, mod = divmod(vdif.stat().st_size, N_BYTES_PER_UNIT)

    if mod:
        raise RuntimeError("The VDIF file is broken.")

    # make binary readers
    read_vdif_head = make_binary_reader(N_ROWS_VDIF_HEAD, UINT)
    read_corr_head = make_binary_reader(N_ROWS_CORR_HEAD, UINT)
    read_corr_data = make_binary_reader(N_ROWS_CORR_DATA, SHORT)

    # prepare an empty zarr dataset
    z = zarr.open(vdif.with_suffix(".zarr").name, mode="w")

    z.empty(
        name="vdif_head",
        shape=(n_units, N_ROWS_VDIF_HEAD),
        chunks=(1, N_ROWS_VDIF_HEAD),
        dtype=np.uint32,
    )
    z.empty(
        name="corr_head",
        shape=(n_units, N_ROWS_CORR_HEAD),
        chunks=(1, N_ROWS_CORR_HEAD),
        dtype=np.uint32,
    )
    z.empty(
        name="corr_data",
        shape=(n_units, N_ROWS_CORR_DATA),
        chunks=(1, N_ROWS_CORR_DATA),
        dtype=np.int16,
    )

    # Read the VDIF file and write it to the Zarr dataset
    with open(vdif, "rb") as f:
        for i in tqdm(range(n_units), disable=not progress):
            z.vdif_head[i] = read_vdif_head(f)
            z.corr_head[i] = read_corr_head(f)
            z.corr_data[i] = read_corr_data(f)

    # return the path of the Zarr dataset
    return Path(z.store.path)


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

