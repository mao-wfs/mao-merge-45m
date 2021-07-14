__all__ = ["convert"]


# standard library
from pathlib import Path
from typing import Optional, Sequence, Union, cast


# third-party packages
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar


# constants
CSV_COLS = "time", "wind_speed", "wind_direction"
JST_HOURS = np.timedelta64(9, "h")


def convert(
    path_csv: Union[Sequence[Path], Path],
    path_zarr: Optional[Path] = None,
    length_per_chunk: int = 1000000,
    overwrite: bool = False,
    progress: bool = False,
) -> Path:
    """Convert a raw CSV file(s) to a formatted Zarr file.

    This function will make a one-dimensional weather log outputs
    with time metadata derived from the raw CSV file.

    Args:
        path_csv: Path(s) of the raw CSV file(s).
        path_zarr: Path of the formatted Zarr file (optional).
        length_per_chunk: Length per chunk in the Zarr file.
        overwrite: Whether to overwrite the formatted Zarr file if exists.
        progress: Whether to show a progress bar.

    Returns:
        Path of the formatted Zarr file.

    Raises:
        FileExistsError: Raised if the formatted Zarr file exists
            and overwriting is not allowed (default).

    Notes:
        The timezone of the Zarr file is not JST but UTC.

    """
    # check the existence of the Zarr file
    if isinstance(path_csv, Path):
        path_csv = [path_csv]

    if path_zarr is None:
        path_zarr = path_csv[0].with_suffix(".zarr")

    if path_zarr.exists() and not overwrite:
        raise FileExistsError(f"{path_zarr} already exists.")

    # read CSV file(s) and convert them to DataFrame(s)
    df = pd.DataFrame(
        columns=CSV_COLS[1:],
        index=pd.Index([], name=CSV_COLS[0]),
    )

    for path in path_csv:
        df_ = pd.read_csv(
            path,
            names=CSV_COLS,
            index_col=0,
            parse_dates=True,
        )
        df = df.append(df_).drop_duplicates()

    # write DataFrame(s) to the Zarr file
    ds = cast(xr.Dataset, df.to_xarray())
    ds = ds.assign_coords(time=ds.time - JST_HOURS)
    ds = ds.chunk(length_per_chunk)

    ds.time.attrs.update(
        long_name="Measured time",
    )
    ds.wind_speed.attrs.update(
        long_name=CSV_COLS[1],
        units="m/s",
    )
    ds.wind_direction.attrs.update(
        long_name=CSV_COLS[2],
        units="degree",
    )

    if progress:
        with ProgressBar():
            ds.to_zarr(path_zarr, mode="w")
    else:
        ds.to_zarr(path_zarr, mode="w")

    return path_zarr
