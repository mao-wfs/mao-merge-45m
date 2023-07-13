# third-party packages
import mao_merge_45m


# constants
AUTHOR: str = "Akio Taniguchi"
VERSION: str = "0.2.0"


# test functions
def test_author() -> None:
    assert mao_merge_45m.__author__ == AUTHOR


def test_version() -> None:
    assert mao_merge_45m.__version__ == VERSION
