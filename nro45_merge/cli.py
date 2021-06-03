# standard library
from logging import basicConfig, DEBUG, getLogger
from typing import Optional


# third-party packages
from fire import Fire


# module-level logger
logger = getLogger(__name__)


# command line interface
def main() -> None:
    """Run command line interface."""
    Fire(merge)


# interface functions
def merge(
    correlator: str,
    accelerometer: Optional[str] = None,
    weather_monitor: Optional[str] = None,
    correlator_log: Optional[str] = None,
    optswitch_log: Optional[str] = None,
    antenna_log: Optional[str] = None,
    output: Optional[str] = None,
    ch_binning: int = 1,
    debug: bool = False,
) -> None:
    """Merge datasets into a single netCDF file.

    Args:
        correlator: Path(s) of correlator data.
        accelerometer: Path of accelerometer data.
        weather_monitor: Path of weather monitor data.
        correlator_log: Path of correlator log.
        optswitch_log: Path of optical switch log.
        antenna_log: Path of antenna log.
        output: Path of output netCDF file.
        ch_binning: Number of channels to bin.
        debug: If True, debug-level log messages are shown.

    Returns:
        This functions returns nothing.

    """
    basicConfig(
        datefmt="%Y-%m-%d %H:%M:%S",
        format="[%(asctime)s %(name)s %(levelname)s]: %(message)s",
    )

    if debug:
        logger.parent.setLevel(DEBUG)

    logger.debug(f"{correlator=}")
    logger.debug(f"{accelerometer=}")
    logger.debug(f"{weather_monitor=}")
    logger.debug(f"{correlator_log=}")
    logger.debug(f"{optswitch_log=}")
    logger.debug(f"{antenna_log=}")
    logger.debug(f"{output=}")
    logger.debug(f"{ch_binning=}")
    logger.debug(f"{debug=}")
