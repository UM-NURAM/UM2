"""The tool that keeps um2.openmc in sync with the pinned OpenMC release."""

from pathlib import Path
import importlib.util
import shutil
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
OPENMC_SRC = REPO / 'extern' / 'openmc'
TOOL = REPO / 'python' / 'tools' / 'sync_openmc.py'

pytestmark = pytest.mark.skipif(not (OPENMC_SRC / 'openmc' / '__init__.py').exists(),
                                reason='extern/openmc submodule not checked out')


@pytest.fixture(scope='module')
def sync():
    spec = importlib.util.spec_from_file_location('sync_openmc', TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules['sync_openmc'] = module
    spec.loader.exec_module(module)
    return module


def test_api_matches_openmc(sync, capsys):
    """Every implemented class, method and function has OpenMC's signature."""
    assert sync.check_api(OPENMC_SRC) == 0
    out = capsys.readouterr().out
    assert '0 error(s)' in out


def test_generated_tables_are_up_to_date(sync, tmp_path):
    """Regenerating the data tables from the submodule reproduces the files
    in the repository."""
    package = REPO / 'python' / 'um2' / 'openmc'
    copy = tmp_path / 'openmc'
    for sub in ('data', 'mgxs'):
        (copy / sub).mkdir(parents=True)
    written = sync.generate(OPENMC_SRC, copy)
    for path in written:
        rel = path.relative_to(copy)
        assert path.read_bytes() == (package / rel).read_bytes(), rel


def test_detects_signature_change(sync, tmp_path):
    """A change of a signature in OpenMC is reported as an error."""
    src = tmp_path / 'openmc_src'
    shutil.copytree(OPENMC_SRC / 'openmc', src / 'openmc')
    material = src / 'openmc' / 'material.py'
    text = material.read_text()
    old = 'def add_nuclide(self, nuclide: str, percent: float, percent_type: str = \'ao\'):'
    assert old in text
    material.write_text(text.replace(
        old, 'def add_nuclide(self, nuclide: str, percent: float, new_arg, '
             'percent_type: str = \'ao\'):'))
    assert sync.check_api(src) > 0
