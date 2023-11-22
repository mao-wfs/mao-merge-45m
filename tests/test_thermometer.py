# standard library
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory


# third-party packages
import pandas as pd
import xarray as xr
from mao_merge_45m.thermometer import convert


# constants
TEST_TXT = Path("data") / "thermometer_20220430T150000.txt"


# test function
def test_convert() -> None:
    """Test whether the data of a txt file is correctly parsed."""
    with TemporaryDirectory() as zarr:
        path_zarr = Path(zarr)
        path_zarr = convert(TEST_TXT, path_zarr, overwrite=True)
        data_ds = xr.open_zarr(path_zarr)

        assert data_ds.time[0] == pd.Timestamp("2022-04-30T15:00:00.000000000")
        assert data_ds.temperature_1p5m[0] == 5.1
        assert data_ds.temperature_30m[0] == 4.8
        assert data_ds.sunshine_flag[0] == False
