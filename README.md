# mao-merge-45m
Merge MAO datasets observed with NRO 45m telescope into a single Zarr file

## Installation

```shell
$ git clone https://github.com/mao-wfs/mao-merge-45m.git
$ cd mao-merge-45m
$ poetry install
```

## Convert raw data to formatted (Zarr) data

### Correlator output

```python
from pathlib import Path
from mao_merge_45m import correlator


path_vdif = Path("/path/to/data.vdif")
path_raw_zarr = correlator.to_zarr(path_vdif)
path_fmt_zarr = correlator.convert(path_raw_zarr)
```

### Accelerometer log

```python
from pathlib import Path
from mao_merge_45m import accelerometer


path_gbd = Path("/path/to/data.gbd")
path_raw_zarr = accelerometer.to_zarr(path_gbd)
path_fmt_zarr = accelerometer.convert(path_raw_zarr)
```

### Weather log

```python
from pathlib import Path
from mao_merge_45m import weather


path_csv = Path("/path/to/data.csv")
path_zarr = accelerometer.convert(path_csv)
```

### Antenna log (normal 10-sps format)

```python
from pathlib import Path
from mao_merge_45m import antenna


path_log = Path("/path/to/data.txt")
path_zarr = antenna.convert(path_log)
```

### Antenna log (new 50-sps format)

```python
from pathlib import Path
from mao_merge_45m import antenna_50_sps


path_log = Path("/path/to/data.txt")
path_zarr = antenna_50_sps.convert(path_log)
```

## Merge formatted data into a single Zarr file

```python
from pathlib import Path
from mao_merge_45m import merge


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
