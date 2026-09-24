"""Acceptance tests: the OpenMC example notebooks, run with um2.openmc.

Each notebook is executed unchanged (``import openmc`` is redirected to
um2.openmc) up to the first cell that needs something outside the scope of
UM2 (e.g. OpenMC simulation results), which must fail with a clear error. The
geometry defined by the notebook is then built, checked against point
sampling, and meshed.

The notebooks are read from $OPENMC_NOTEBOOKS, or a checkout of
https://github.com/openmc-dev/openmc-notebooks next to the UM2 repository.
"""

import math
import os

import pytest

from notebook_runner import notebooks_dir, run_notebook
from conftest import requires_core

pytestmark = pytest.mark.skipif(not notebooks_dir().is_dir(),
                                reason='openmc-notebooks checkout not found')
os.environ.setdefault('MPLBACKEND', 'Agg')

# notebook: (index of the first cell that fails, expected exception, message)
STOPS = {
    'pincell': (33, 'UM2NotSupportedError', 'Sphere'),
    'hexagonal-lattice': (None, None, None),
    'mg-mode-part-i': (32, 'UM2NotSupportedError', 'StatePoint'),
    'mg-mode-part-ii': (39, 'UM2NotSupportedError', 'mgxs.Library'),
    'mg-mode-part-iii': (39, 'UM2NotSupportedError', 'mgxs.Library'),
    'candu': (26, 'UM2NotSupportedError', 'StatePoint'),
    'search': (5, 'UM2NotSupportedError', 'search_for_keff'),
    'depletion': (20, 'UM2NotSupportedError', 'deplete'),
}

# notebook: (cells to run, cells to skip, variable holding the geometry/model)
GEOMETRY = {
    'pincell': (71, range(33, 56), 'geometry'),
    'hexagonal-lattice': (None, (), 'geometry'),
    'mg-mode-part-i': (25, (), 'model'),
    'mg-mode-part-ii': (24, (), 'model'),
    'mg-mode-part-iii': (24, (), 'model'),
    'candu': (20, (), 'geometry'),
}


@pytest.mark.parametrize('name', sorted(STOPS))
def test_notebook_runs_unchanged(name, tmp_path):
    stop_cell, exc, message = STOPS[name]
    if stop_cell is None:
        run_notebook(name, workdir=tmp_path)
        return
    with pytest.raises(Exception) as info:
        run_notebook(name, workdir=tmp_path)
    assert type(info.value).__name__ == exc
    assert message in str(info.value)
    frames = [f for f in info.traceback if str(f.path).startswith(f'{name}.ipynb')]
    assert frames and str(frames[-1].path) == f'{name}.ipynb[{stop_cell}]'


def notebook_geometry(name, tmp_path):
    import um2.openmc as openmc
    stop, skip, var = GEOMETRY[name]
    ns = run_notebook(name, stop=stop, skip=skip, workdir=tmp_path)
    obj = ns[var]
    geometry = obj.geometry if isinstance(obj, openmc.Model) else obj
    return obj, geometry


@requires_core
@pytest.mark.parametrize('name', [
    'pincell', 'hexagonal-lattice', 'mg-mode-part-i', 'mg-mode-part-ii',
    'candu', pytest.param('mg-mode-part-iii', marks=pytest.mark.slow)])
def test_notebook_geometry(name, tmp_path, gmsh_session):
    import um2
    import um2.openmc as openmc
    from test_build import check_against_sampling
    obj, geometry = notebook_geometry(name, tmp_path)
    result = um2.build_geometry(geometry)
    assert result.material_surfaces
    check_against_sampling(geometry, result)

    gmsh_session.model.add('mesh')
    kwargs = {}
    if isinstance(obj, openmc.Model) and obj.materials.cross_sections is not None:
        # Resolve the library relative to the notebook directory
        kwargs = dict(cross_sections=notebooks_dir() / obj.materials.cross_sections.name,
                      kn_target=12.0)
    mesh = um2.mesh_geometry(obj, mesh_type='Tri', output=tmp_path / 'model.xdmf',
                             **kwargs)
    assert mesh.model.num_fine_cells_total > 0
    volumes = dict(line.split() for line in
                   (tmp_path / 'model.volumes').read_text().splitlines())
    total = mesh.plan.size[0] * mesh.plan.size[1]
    assert sum(map(float, volumes.values())) == pytest.approx(total, rel=1e-6)


@requires_core
def test_c5g7_notebook_partition(tmp_path, gmsh_session):
    """mg-mode-part-i: the 2D C5G7 core is inferred as a 3 x 3 core of 17 x 17
    pin lattices with 7 unique pins."""
    import um2
    obj, geometry = notebook_geometry('mg-mode-part-i', tmp_path)
    plan = um2.mpact.infer_hierarchy(geometry)
    assert plan.core.shape == (3, 3)
    assert len(plan.assemblies) == 3
    assert len(plan.coarse_cells) == 7
    assert plan.size == pytest.approx((64.26, 64.26))
    fuel = [c for c in plan.coarse_cells if 'water' not in c.label]
    assert len(fuel) == 6


@requires_core
def test_hexagonal_notebook_areas(tmp_path, gmsh_session):
    """hexagonal-lattice: exact areas of the final (x-oriented, hexagonal
    prism) geometry."""
    import um2
    _, geometry = notebook_geometry('hexagonal-lattice', tmp_path)
    areas = um2.build_geometry(geometry).material_areas()
    assert areas['fuel'] == pytest.approx(33 * math.pi * 0.25**2, rel=1e-9)
    assert areas['fuel2'] == pytest.approx(4 * math.pi * 0.5**2, rel=1e-9)
    edge = 4 * 1.25
    assert sum(areas.values()) == pytest.approx(1.5 * math.sqrt(3) * edge**2, rel=1e-9)
