# standard library
from pathlib import Path


# third-party packages
import pandas as pd
from nro45_merge.accelerometer import get_data, get_header


# constants
TEST_GBD = Path("data") / "2018-12-02_09_58_42.gbd"


# test functions
def test_get_header():
    """Test whether the header of a GBD file is correctly parsed."""
    header = get_header(TEST_GBD)

    assert header["Common"]["ID"] == "3502111C"  # first item
    assert header["Waveform"]["XYLine"]["CH4"] == [21, 0, 21]  # last item


def test_get_data():
    """Test whether the data of a GBD file is correctly parsed."""
    data = get_data(TEST_GBD)

    assert data.columns[0] == "CH1 (mV)"
    assert data.columns[-1] == "Status (None)"
    assert data.index[0] == pd.Timestamp("2018-12-02 09:58:42.000000")
    assert data.index[-1] == pd.Timestamp("2018-12-02 10:28:42.010000")
