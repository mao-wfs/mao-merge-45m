# standard library
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory


# third-party packages
import pandas as pd
import xarray as xr
from mao_merge_45m.power_meter import convert


# constants
TEST_CSV = Path("data") / "power_meter_2022144230000Z.csv"


# test function
def test_convert() -> None:
    """Test whether the data of a CSV file is correctly parsed."""
    with TemporaryDirectory() as zarr:
        path_zarr = Path(zarr)
        path_zarr = convert(TEST_CSV, path_zarr, overwrite=True)
        data_ds = xr.open_zarr(path_zarr)

        assert data_ds.time[0] == pd.Timestamp("2022-05-24T22:58:49.000000000")
        assert data_ds.power_meter[0] == -14.161

