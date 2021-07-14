# standard library
from pathlib import Path


# third-party packages
import pandas as pd
from nro45_merge.accelerometer import get_data, get_header


# constants
TEST_GBD = Path("data") / "accelerometer_20201123T004000Z.gbd"


# test functions
def test_get_header():
    """Test whether the header of a GBD file is correctly parsed."""
    header = get_header(TEST_GBD)

    assert header["Common"]["ID"] == 44171012  # first item
    assert header["Waveform"]["XYLine"]["CH4"] == [21, 0, 21]  # last item


def test_get_data():
    """Test whether the data of a GBD file is correctly parsed."""
    data = get_data(TEST_GBD)

    assert data.columns[0] == "CH1 (V)"
    assert data.columns[-1] == "Status (None)"
    assert data.index[0] == pd.Timestamp("2020-11-23 09:40:00.000000")
    assert data.index[-1] == pd.Timestamp("2020-11-23 09:44:59.990000")
    assert data["CH1 (V)"][0] == -0.350
    assert data["CH1 (V)"][-1] == 1.300
