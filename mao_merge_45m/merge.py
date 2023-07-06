__all__ = ["merge"]


# standard library
from pathlib import Path
from typing import Optional

# third-party packages
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


# main features
def merge(
    path_correlator_zarr: Path,
    path_merged_zarr: Optional[Path] = None,
    *,
    path_accelerometer_zarr: Optional[Path] = None,
    path_weather_zarr: Optional[Path] = None,
    path_antenna_zarr: Optional[Path] = None,
    path_sam45_zarr: Optional[Path] = None,
    interpolation: str = "linear",
    correlator_time_offset: int = 0,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Merge Zarr files of measured data into a single Zarr file.

    Args:
        path_correlator_zarr: Path of the correlator Zarr file.
        path_merged_zarr: Path of the merged Zarr file.
        path_accelerometer_zarr: Path of the accelerometer Zarr file.
        path_weather_zarr: Path of the weather Zarr file.
        path_antenna_zarr: Path of the antenna Zarr file.
        path_sam45_zarr: Path of the SAM45 Zarr file.
        interpolation: Method of interpolation of log data.
        correlator_time_offset: Offset time in units of ms to add to correlator data
        overwrite: Whether to overwrite the merged Zarr file if exists.
        progress: Whether to show a progress bar.

    Returns:
        Path of the merged Zarr file.

    Raises:
        FileExistsError: Raised if the merged Zarr file exists
            and overwriting is not allowed (default).

    """
    if path_merged_zarr is None:
        path_merged_zarr = path_correlator_zarr.with_suffix(".merged.zarr")

    if path_merged_zarr.exists() and not overwrite:
        raise FileExistsError(f"{path_merged_zarr} already exists.")

    # open correlator Zarr and correct time offset (if any)
    correlator = xr.open_zarr(path_correlator_zarr)
    correlator.coords["time"] = correlator.coords["time"] + np.timedelta64(
        correlator_time_offset, "ms"
    )

    # append metadata Zarrs to the correlator Zarr
    for path in (
        path_accelerometer_zarr,
        path_weather_zarr,
        path_antenna_zarr,
        path_sam45_zarr,
    ):
        if path is None:
            continue

        ds = xr.open_zarr(path).interp_like(correlator, interpolation)
        correlator.coords.update(ds)

    if progress:
        with ProgressBar():
            correlator.to_zarr(path_merged_zarr, mode="a")
    else:
        correlator.to_zarr(path_merged_zarr, mode="a")

    return path_merged_zarr
