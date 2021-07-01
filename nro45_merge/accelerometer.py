# standard library
import re
from pathlib import Path
from typing import Any, Union, Dict, List


# third-party packages
from tomlkit import parse


# constants
HEADER_REMOVAL = re.compile("\x00| ")
HEADER_SIZE = re.compile("HeaderSiz\s+=\s+(\d+)")
HEADER_SECTION = re.compile("^(\$+)(.+)$")
HEADER_VALUE = re.compile("^\s*(\w+)\s*=\s*(.+)$")


# main features
def get_header(path: Path, encoding: str = "shift-jis") -> Dict[str, Any]:
    """Return the header of a GBD file as a dictionary."""
    size = get_header_size(path, encoding)

    with path.open("rb") as f:
        header = f.read(size).decode(encoding)
        header_rows = HEADER_REMOVAL.sub("", header).split()

    toml_rows = []
    section = ["", "", ""]

    for row in header_rows:
        toml_rows.append(parse_header_row(row, section))

    return parse("\n".join(toml_rows))


# helper features
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
