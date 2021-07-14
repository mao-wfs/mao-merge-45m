# standard library
from pathlib import Path
from tempfile import TemporaryDirectory


# third-party packages
import numpy as np
import pandas as pd
import xarray as xr
from nro45_merge.weather import convert


# constants
CSV_COLS = "time", "wind_speed", "wind_direction"
TEST_CSV = Path("data") / "weather_20201123T004000Z.csv"
JST_HOURS = np.timedelta64(9, "h")


# test functions
def test_convert():
    """Test whether a CSV file is correctly parsed."""
    with TemporaryDirectory() as zarr:
        path_zarr = Path(zarr)
        convert(TEST_CSV, path_zarr, overwrite=True)

        ds = xr.open_zarr(path_zarr)
        ds = ds.assign_coords(time=ds.time + JST_HOURS)

        df = pd.read_csv(
            TEST_CSV,
            names=CSV_COLS,
            index_col=0,
            parse_dates=True,
        )

        assert (ds.wind_speed.to_series() == df[CSV_COLS[1]]).all()
        assert (ds.wind_direction.to_series() == df[CSV_COLS[2]]).all()
