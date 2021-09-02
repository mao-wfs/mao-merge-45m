# third-party packages
from typing_extensions import Literal
from xarray_dataclasses import AsDataArray, AsDataset, Attr, Data, Dataof


# type hints
T = Literal["time"]


# dataclasses
class Azimuth(AsDataArray):
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Azimuth"
    units: Attr[str] = "degree"


class Elevation(AsDataArray):
    data: Data[T, float] = 0.0
    long_name: Attr[str] = "Elevation"
    units: Attr[str] = "degree"


class Antenna(AsDataset):
    """Representation of antenna logs in xarray."""

    azimuth: Dataof[Azimuth] = 0.0
    """Azimuth of an antenna."""

    elevation: Dataof[Elevation] = 0.0
    """Elevation of an antenna."""
