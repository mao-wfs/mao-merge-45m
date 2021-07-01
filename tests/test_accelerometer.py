# standard library
from pathlib import Path


# third-party packages
from nro45_merge.accelerometer import get_header


# constants
TEST_GBD = Path("data") / "2018-12-02_09_58_42.gbd"


# test functions
def test_get_header():
    """Test whether the header of a GBD file is correctly parsed."""
    header = get_header(TEST_GBD)

    assert header["Common"]["ID"] == "3502111C"  # first item
    assert header["Waveform"]["XYLine"]["CH4"] == [21, 0, 21]  # last item
