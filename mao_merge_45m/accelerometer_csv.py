# standard library
from dataclasses import dataclass
from typing import Literal


# dependencies
from xarray_dataclasses import AsDataset, Attr, Data, Dataof


# type hints
Time = Literal["time"]


@dataclass
class CH1:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH1"
    units: Attr[str] = "mV"


@dataclass
class CH2:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH2"
    units: Attr[str] = "mV"


@dataclass
class CH3:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH3"
    units: Attr[str] = "mV"


@dataclass
class CH4:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH4"
    units: Attr[str] = "mV"


@dataclass
class CH5:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH5"
    units: Attr[str] = "mV"


@dataclass
class CH6:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH6"
    units: Attr[str] = "mV"


@dataclass
class CH7:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH7"
    units: Attr[str] = "degC"


@dataclass
class CH8:
    data: Data[Time, float]
    long_name: Attr[str] = "Accelerometer CH8"
    units: Attr[str] = "degC"


@dataclass
class Accelerometer(AsDataset):
    """Specification of accelerometer logs in xarray."""

    accelerometer_ch1: Dataof[CH1]
    """Data of CH1."""

    accelerometer_ch2: Dataof[CH2]
    """Data of CH2."""

    accelerometer_ch3: Dataof[CH3]
    """Data of CH3."""

    accelerometer_ch4: Dataof[CH4]
    """Data of CH4."""

    accelerometer_ch5: Dataof[CH5]
    """Data of CH5."""

    accelerometer_ch6: Dataof[CH6]
    """Data of CH6."""

    accelerometer_ch7: Dataof[CH7]
    """Data of CH7."""

    accelerometer_ch8: Dataof[CH8]
    """Data of CH8."""
