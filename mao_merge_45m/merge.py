__all__ = ["merge"]


# standard library
from pathlib import Path
from typing import Optional


# third-party packages
import xarray as xr
from dask.diagnostics import ProgressBar


# main features
def merge(
    path_correlator_zarr: Path,
    path_accelerometer_zarr: Optional[Path] = None,
    path_weather_zarr: Optional[Path] = None,
    path_antenna_zarr: Optional[Path] = None,
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
        path_antenna_zarr: Path of the antenna Zarr file.
        path_merge_zarr: Path of the merge Zarr file.
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

    # create (overwrite) the merge Zarr
    correlator = xr.open_zarr(path_correlator_zarr)

    if not in_place:
        correlator.to_zarr(path_merge_zarr, mode="w")

    # append the other Zarrs to the merge Zarr
    for path in (
        path_accelerometer_zarr,
        path_weather_zarr,
        path_antenna_zarr,
    ):
        if path is None:
            continue

        ds = xr.Dataset(coords=xr.open_zarr(path).variables)
        ds = ds.interp_like(correlator, interpolation)

        if progress:
            with ProgressBar():
                ds.to_zarr(path_merge_zarr, mode="a")
        else:
            ds.to_zarr(path_merge_zarr, mode="a")

    return path_merge_zarr
