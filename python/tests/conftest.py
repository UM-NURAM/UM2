import os
from pathlib import Path
import sys
import warnings

import pytest

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parents[1]
sys.path.insert(0, str(TESTS))

import um2.openmc as openmc  # noqa: E402

C5G7_XSL = REPO / 'tests' / 'physics' / 'xs_files' / 'c5g7.xsl'


def _has_core():
    try:
        import um2._um2  # noqa: F401
        import gmsh  # noqa: F401
    except ImportError:
        return False
    return True


HAS_CORE = _has_core()
requires_core = pytest.mark.skipif(not HAS_CORE,
                                   reason='um2._um2 (C++ bindings) not built')


@pytest.fixture(autouse=True)
def reset_ids():
    openmc.reset_auto_ids()
    yield


@pytest.fixture
def run_in_tmpdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def gmsh_session():
    """An initialized UM2/gmsh session with a fresh gmsh model."""
    if not HAS_CORE:
        pytest.skip('um2._um2 (C++ bindings) not built')
    import gmsh
    import um2
    um2.initialize(log_level=1)
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.model.add(f'test{os.getpid()}')
    yield gmsh
    gmsh.clear()


@pytest.fixture(scope='session')
def c5g7_library():
    from um2 import xs
    return xs.read_xsl(C5G7_XSL)
