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
    path_merged_zarr: Optional[Path] = None,
    *,
    path_accelerometer_zarr: Optional[Path] = None,
    path_weather_zarr: Optional[Path] = None,
    path_antenna_zarr: Optional[Path] = None,
    path_sam45_zarr: Optional[Path] = None,
    interpolation: str = "linear",
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

    # create (overwrite) the merged Zarr
    correlator = xr.open_zarr(path_correlator_zarr)
    correlator.to_zarr(path_merged_zarr, mode="w")

    # append the other Zarrs to the merged Zarr
    for path in (
        path_accelerometer_zarr,
        path_weather_zarr,
        path_antenna_zarr,
        path_sam45_zarr,
    ):
        if path is None:
            continue

        ds = xr.Dataset(coords=xr.open_zarr(path).variables)
        ds = ds.interp_like(correlator, interpolation)

        if progress:
            with ProgressBar():
                ds.to_zarr(path_merged_zarr, mode="a")
        else:
            ds.to_zarr(path_merged_zarr, mode="a")

    return path_merged_zarr
