# standard library
from pathlib import Path
from typing import Optional


# third-party packages
import xarray as xr
from dask.diagnostics import ProgressBar


# main features
def merge(
    path_correlator_zarr: Path,
    path_accelerometer_zarr: Path,
    path_weather_zarr: Path,
    path_merge_zarr: Optional[Path] = None,
    interpolation: str = "linear",
    in_place: bool = False,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Merge Zarr files of measured data into a single Zarr file.

    Args:
        path_correlator_zarr: Path of the correlator Zarr file.
        path_accelerometer_zarr: Path of the accelerometer Zarr file.
        path_weather_zarr: Path of the weather Zarr file.
        path_merge_zarr: Path of the merge Zarr file (optional).
        interpolation: Method of interpolation of log data.
        in_place: When True, log data are added to the correlator Zarr file.
        overwrite: Whether to overwrite the merge Zarr file if exists.
        progress: Whether to show a progress bar.

    Returns:
        Path of the formatted Zarr file.

    Raises:
        FileExistsError: Raised if the formatted Zarr file exists
            and overwriting is not allowed (default).

    """
    # check the existence of the merge Zarr file
    if in_place:
        path_merge_zarr = path_correlator_zarr

    if path_merge_zarr is None:
        raise ValueError("Path of merge Zarr file is not specified.")

    if not in_place and not overwrite and path_merge_zarr.exists():
        raise FileExistsError(f"{path_merge_zarr} already exists.")

    # open all Zarr files
    correlator = xr.open_zarr(path_correlator_zarr)
    accelerometer = xr.open_zarr(path_accelerometer_zarr)
    weather = xr.open_zarr(path_weather_zarr)

    # move data variables to coordinates
    accelerometer = xr.Dataset(coords=accelerometer.variables)
    weather = xr.Dataset(coords=weather.variables)

    # interpolate log data to fit the time axis of the correlator
    accelerometer = accelerometer.interp_like(correlator, interpolation)
    weather = weather.interp_like(correlator, interpolation)

    # append log data to the correlator or create the merge Zarr file
    if progress:
        bar = ProgressBar()
    else:
        bar = ProgressBar(float("inf"))

    with bar:
        if in_place:
            accelerometer = accelerometer.merge(weather)
            accelerometer.to_zarr(path_correlator_zarr, mode="a")
        else:
            correlator = correlator.merge(accelerometer)
            correlator = correlator.merge(weather)
            correlator.to_zarr(path_merge_zarr)

    return path_merge_zarr
