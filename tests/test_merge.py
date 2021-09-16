# standard library
from pathlib import Path
from tempfile import TemporaryDirectory


# third-party packages
import xarray as xr
from nro45_merge import accelerometer, antenna, correlator, merge, weather


# constants
TEST_CSV = Path("data") / "weather_20201123T004000Z.csv"
TEST_GBD = Path("data") / "accelerometer_20201123T004000Z.gbd"
TEST_LOG = Path("data") / "antenna_20201123T004100Z.txt"
TEST_VDIF = Path("data") / "correlator_20201123T004100Z.vdif"
TEST_DIST = Path("data") / "distribution_20201123T004100Z.zarr"


# test functions
def test_merge() -> None:
    """Test whether a merged Zarr file is identical to the test file."""
    with TemporaryDirectory() as d:
        path_dir = Path(d)

        # correlator (VDIF -> Zarr)
        path_corr = correlator.convert(
            correlator.to_zarr(
                TEST_VDIF,
                path_dir / "correlator.zarr",
                seconds_per_chunk=1,
            )
        )

        # accelerometer (GBD -> Zarr)
        path_acc = accelerometer.convert(
            accelerometer.to_zarr(
                TEST_GBD,
                path_dir / "accelerometer.zarr",
            )
        )

        # weather (CSV -> Zarr)
        path_wea = weather.convert(
            TEST_CSV,
            path_dir / "weather.zarr",
        )

        # antenna (log -> Zarr)
        path_ant = antenna.convert(
            TEST_LOG,
            path_dir / "antenna.zarr",
        )

        # merge (Zarrs -> Zarr)
        path_fmt = merge.merge(
            path_corr,
            path_acc,
            path_wea,
            path_ant,
            path_dir / "distribution.zarr",
        )

        # comparison
        ds_fmt = xr.open_zarr(path_fmt)
        ds_test = xr.open_zarr(TEST_DIST)
        assert (ds_fmt == ds_test).all()
