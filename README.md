# nro45-merge
Merge MAO datasets observed with NRO 45m telescope into a single Zarr file

## Installation

```shell
$ git clone https://github.com/mao-wfs/nro45-merge.git
$ cd nro45-merge
$ poetry install
```

## Convert raw data to formatted (Zarr) data

### Correlator output

```python
from pathlib import Path
from nro45_merge import correlator


path_vdif = Path("/path/to/data.vdif")
path_raw_zarr = correlator.to_zarr(path_vdif)
path_fmt_zarr = correlator.convert(path_raw_zarr)
```

### Accelerometer log

```python
from pathlib import Path
from nro45_merge import accelerometer


path_gbd = Path("/path/to/data.gbd")
path_raw_zarr = accelerometer.to_zarr(path_gbd)
path_fmt_zarr = accelerometer.convert(path_raw_zarr)
```

### Weather log

```python
from pathlib import Path
from nro45_merge import weather


path_csv = Path("/path/to/data.csv")
path_zarr = accelerometer.convert(path_csv)
```

## Merge formatted data into a single Zarr file

```python
from pathlib import Path
from nro45_merge import merge


path_correlator = Path("/path/to/formatted/correlator.zarr")
path_accelerometer = Path("/path/to/formatted/accelerometer.zarr")
path_weather = Path("/path/to/formatted/weather.zarr")
path_merged = Path("/path/to/merged.zarr")


merge.merge(
    path_correlator,
    path_accelerometer,
    path_weather,
    path_merged,
)
```
