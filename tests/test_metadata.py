# third-party packages
import nro45_merge


# constants
AUTHOR: str = "Akio Taniguchi"
VERSION: str = "0.1.0"


# test functions
def test_author() -> None:
    assert nro45_merge.__author__ == AUTHOR


def test_version() -> None:
    assert nro45_merge.__version__ == VERSION
