# standard library
from pathlib import Path
from tempfile import TemporaryDirectory


# third-party packages
import numpy as np
import pandas as pd
import xarray as xr
from nro45_merge.antenna import convert


# constants
DATE_FORMAT = "%y%m%d%H%M%S"
JST_HOURS = np.timedelta64(9, "h")
LOG_COLS = "time", "antenna_azimuth", "antenna_elevation"
TEST_LOG = Path("data") / "antenna_20201123T004100Z.txt"


# test functions
def test_convert() -> None:
    """Test whether a log file is correctly parsed."""
    with TemporaryDirectory() as zarr:
        path_zarr = Path(zarr)
        convert(TEST_LOG, path_zarr, overwrite=True)

        ds = xr.open_zarr(path_zarr)
        ds = ds.assign_coords(time=ds.time + JST_HOURS)

        df = pd.read_csv(
            TEST_LOG,
            header=0,
            index_col=0,
            names=LOG_COLS,
            sep=r"\s+",
            usecols=range(len(LOG_COLS)),
        )

        index = pd.to_datetime(df.index, format=DATE_FORMAT)
        df.set_index(index, inplace=True)

        assert (ds.antenna_azimuth.to_series() == df[LOG_COLS[1]]).all()
        assert (ds.antenna_elevation.to_series() == df[LOG_COLS[2]]).all()
