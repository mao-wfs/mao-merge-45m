# standard library
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory


# third-party packages
from mao_merge_45m.correlator import to_vdif, to_zarr


# constants
TEST_VDIF = Path("data") / "correlator_20201123T004100Z.vdif"


# test functions
def test_zarr_conversion() -> None:
    """Test whether a VDIF file and that regenerated are identical."""
    with TemporaryDirectory() as zarr, NamedTemporaryFile() as vdif:
        path_zarr = Path(zarr)
        path_vdif = Path(vdif.name)

        to_zarr(TEST_VDIF, path_zarr, overwrite=True, seconds_per_chunk=1)
        to_vdif(path_zarr, path_vdif, overwrite=True)

        with TEST_VDIF.open("rb") as f, path_vdif.open("rb") as g:
            assert f.read() == g.read()
