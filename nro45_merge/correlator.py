# standard library
from logging import getLogger
from struct import Struct
from typing import BinaryIO, Callable, Tuple


# module-level logger
logger = getLogger(__name__)


# constants
LITTLE_ENDIAN = "<"


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

